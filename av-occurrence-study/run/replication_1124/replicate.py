"""Freeze, run and summarize a controlled option/evidence-label test."""
import argparse
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest"
sys.path.insert(0, str(HERE.parent))
from infer_mlx import decode, load_manifest, prompt_for

CONDITIONS = ("original", "options_reversed", "segment_labels_reversed")
MODEL_REVISION = "fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b"
SHARD_SHA = {
    "model-00001-of-00002.safetensors": "f80072ac0e82a2ace0740eab995f6f9ae2623adb55e6906e21c2f8edca537803",
    "model-00002-of-00002.safetensors": "016bd9d3596e9b824f5a527875e5830414552c950eee794e98efb44c1cd8dc6a",
}


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, items):
    with path.open("x", encoding="utf-8") as stream:
        for row in items:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def content_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze():
    selected = read(MANIFEST / "selected.jsonl")
    prompts, media_hashes = [], {}
    for candidate in selected:
        pair = candidate["pair_id"]
        frames = MANIFEST / "frames" / pair
        subprocess.run([
            sys.executable, str(HERE.parent / "prepare_epic_smoke.py"),
            "--queue", str(MANIFEST / "selected.jsonl"),
            "--media-manifest", str(MANIFEST / "media.jsonl"),
            "--output-dir", str(frames), "--pair", pair,
        ], check=True)
        questions = load_manifest(frames / "smoke-input.jsonl")
        for ordinal, row in enumerate(questions, 1):
            prompt, images, options, segments = prompt_for(row, "flat")
            if len(images) != 6 or len(options) != 2 or len(segments) != 2:
                raise ValueError("Experiment requires six frames, two options and two time windows")
            image_paths = [str(Path(image).relative_to(MANIFEST)) for image in images]
            for image, relative in zip(images, image_paths):
                media_hashes[relative] = content_hash(Path(image))
            common = {
                "qid": row["qid"], "pair_id": pair, "video_id": candidate["video_id"],
                "ordinal": ordinal, "frames": image_paths,
                "physical_window_order": list(segments.values()), "semantic_actions": row["options"],
            }
            prompts.append(dict(common, condition="original", prompt=prompt,
                                option_codes=options, segment_codes=segments))
            reversed_row = dict(row, options=dict(reversed(list(row["options"].items()))))
            changed, changed_images, changed_options, unchanged_segments = prompt_for(reversed_row, "flat")
            old_choices = "\n".join(f"{key}. {row['options'][value]}" for key, value in options.items())
            new_choices = "\n".join(f"{key}. {row['options'][value]}" for key, value in changed_options.items())
            if changed != prompt.replace(old_choices, new_choices, 1) or changed_images != images or unchanged_segments != segments:
                raise ValueError("Option reversal also changed an unrelated input")
            prompts.append(dict(common, condition="options_reversed", prompt=changed,
                                option_codes=changed_options, segment_codes=segments))
            reverse_ids = {"E01": "E02", "E02": "E01"}
            changed = re.sub(r"(?m)^E0[12](?= )", lambda match: reverse_ids[match[0]], prompt)
            if changed == prompt or len(re.findall(r"(?m)^E0[12](?= )", changed)) != 4:
                raise ValueError("Unexpected evidence format")
            prompts.append(dict(common, condition="segment_labels_reversed", prompt=changed,
                                option_codes=options, segment_codes={reverse_ids[k]: v for k,v in segments.items()}))
    prompt_file = MANIFEST / "prompts.jsonl"
    write(prompt_file, prompts)
    with (MANIFEST / "input_hashes.json").open("x") as stream:
        json.dump({"prompts_sha256": content_hash(prompt_file), "frame_sha256": media_hashes,
                   "runner_sha256": content_hash(Path(__file__)), "checkpoint_revision": MODEL_REVISION,
                   "condition_order": CONDITIONS}, stream, indent=2)
    print(f"Frozen {len(prompts)} input rows across {len(selected)} videos", flush=True)


def run(checkpoint):
    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    frozen = json.loads((MANIFEST / "input_hashes.json").read_text())
    prompt_file = MANIFEST / "prompts.jsonl"
    if content_hash(Path(__file__)) != frozen["runner_sha256"]:
        raise ValueError("Runner changed after freezing the inputs")
    if content_hash(prompt_file) != frozen["prompts_sha256"]:
        raise ValueError("Prompt manifest differs from its frozen fingerprint")
    for path, checksum in frozen["frame_sha256"].items():
        if content_hash(MANIFEST / path) != checksum:
            raise ValueError(f"Frame differs from frozen fingerprint: {path}")
    for name, checksum in SHARD_SHA.items():
        if content_hash(checkpoint / name) != checksum:
            raise ValueError(f"Incorrect checkpoint: {name}")
    model_path = str(checkpoint)
    started = perf_counter()
    model, processor = load(model_path)
    config = load_config(model_path)
    load_seconds = round(perf_counter() - started, 3)
    print(f"Model loaded in {load_seconds}s", flush=True)
    with (HERE / "predictions.jsonl").open("x") as output:
        for item in read(prompt_file):
            formatted = apply_chat_template(processor, config, item["prompt"], num_images=len(item["frames"]))
            mx.reset_peak_memory()
            started = perf_counter()
            generated = generate(model, processor, formatted, image=[str(MANIFEST / p) for p in item["frames"]],
                                 temperature=0.0, max_tokens=64, verbose=False)
            elapsed = round(perf_counter()-started, 3)
            answer, interval, parse_status = decode(generated.text, item["option_codes"], item["segment_codes"])
            row = {k: item[k] for k in ("video_id", "pair_id", "qid", "ordinal", "condition")}
            row.update(answer_id=answer, action=item["semantic_actions"].get(answer), interval_s=interval,
                       physical_window=item["physical_window_order"].index(interval)+1 if interval is not None else None,
                       result_format=parse_status, raw_response=generated.text, prompt_tokens=generated.prompt_tokens,
                       latency_s=elapsed, peak_memory_gb=generated.peak_memory, generated_tokens=generated.generation_tokens,
                       mlx_version=version("mlx"), mlx_vlm_version=version("mlx-vlm"), load_s=load_seconds)
            output.write(json.dumps(row, ensure_ascii=False)+"\n")
            output.flush()
            print(row["qid"],row["condition"],row["action"],row["physical_window"],row["result_format"],flush=True)


def summarize():
    from csv import DictWriter
    rows = read(HERE / "predictions.jsonl")
    expected = read(MANIFEST / "prompts.jsonl")
    indexed = defaultdict(dict)
    expected_ids = {(row["qid"], row["condition"]) for row in expected}
    observed_ids = {(row["qid"], row["condition"]) for row in rows}
    if expected_ids != observed_ids or len(observed_ids) != len(rows):
        raise ValueError("Predictions do not cover the frozen rows exactly once")
    for item in rows:
        indexed[item["qid"]][item["condition"]] = item
    comparisons = []
    summary = {condition: Counter() for condition in CONDITIONS[1:]}
    for qid, group in indexed.items():
        original = group["original"]
        for condition in CONDITIONS[1:]:
            test = group[condition]
            available = all(row["result_format"] == "ok" and row["answer_id"] is not None and row["physical_window"] is not None
                            for row in (original,test))
            action_same = original["answer_id"] == test["answer_id"] if available else None
            physical_changed = original["physical_window"] != test["physical_window"] if available else None
            joint = action_same and physical_changed if available else None
            row = {
                "qid": qid, "video_id": original["video_id"], "ordinal": original["ordinal"],
                "intervention": condition, "original_action": original["action"], "test_action": test["action"],
                "original_window": original["physical_window"], "test_window": test["physical_window"],
                "comparable": available, "action_stable": action_same, "physical_window_changed": physical_changed,
                "action_stable_and_window_changed": joint, "original_prompt_tokens": original["prompt_tokens"],
                "test_prompt_tokens": test["prompt_tokens"],
            }
            comparisons.append(row)
            summary[condition].update(all_questions=1, comparable_questions=int(available),
                 action_stable=int(bool(action_same)), physical_window_changed=int(bool(physical_changed)),
                 action_stable_and_window_changed=int(bool(joint)))
    with (HERE / "comparisons.csv").open("x", newline="") as handle:
        writer = DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    result = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    with (HERE / "summary.json").open("x") as stream:
        stream.write(result)
    print(result, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("freeze","run","summarize"))
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    if args.step == "freeze":
        freeze()
    elif args.step == "run":
        if args.model is None:
            parser.error("run requires --model")
        run(args.model.resolve())
    else:
        summarize()
