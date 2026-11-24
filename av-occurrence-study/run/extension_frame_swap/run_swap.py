"""Generate first replies for the frozen frame-group swap in a new output directory."""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "participant_extension"
sys.path.insert(0, str(BASE / "results_1124"))
from run_generation import check_checkpoint, check_frames, frozen


def read_rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def write(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = HERE / "planned.jsonl"
    if frozen.file_sha256(plan) != "5ff907df5432b5230963e604fd5e4daad86b0edd4a122756a975e50ab545aba2":
        raise ValueError("Plan fingerprint changed")
    rows = read_rows(plan)
    prompts = read_rows(BASE / "frozen_inputs/prompts.jsonl")
    original = {r["qid"]: r for r in read_rows(BASE / "results_1124/replies.jsonl") if r["condition"] == "original"}
    frame_hashes = {}
    for row in rows:
        if row["status"] != "runnable":
            continue
        p = prompts[row["prompt_line"] - 1]
        if (p["qid"], p["condition"], p["frames"][3:] + p["frames"][:3]) != (row["qid"], "original", row["frames"]):
            raise ValueError("Unexpected prompt or frame order for " + row["qid"])
        if hashlib.sha256(p["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
            raise ValueError("Mismatching prompt digest")
        frame_hashes.update(zip(row["frames"], row["frames_sha256"], strict=True))
    check_frames(args.frames_root, frame_hashes)
    checkpoint = check_checkpoint(args.model)
    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    config = load_config(str(args.model))
    model, processor = load(str(args.model))
    args.output_dir.mkdir()
    with (args.output_dir / "attempts.jsonl").open("x") as attempts, (args.output_dir / "replies.jsonl").open("x") as replies:
        write(attempts, dict(event="run_started", at=stamp(), plan_sha256=frozen.file_sha256(plan),
                             runner_sha256=frozen.file_sha256(Path(__file__)), checkpoint_sha256=checkpoint,
                             model_revision=frozen.MODEL_REVISION, python=platform.python_version(),
                             mlx=version("mlx"), mlx_vlm=version("mlx-vlm"), device=str(mx.default_device()),
                             temperature=frozen.TEMPERATURE, max_tokens=frozen.MAX_TOKENS, frame_sha256=frame_hashes))
        for row in rows:
            key = {k: row[k] for k in ("qid", "video_id", "condition", "ordinal", "source_run_order")}
            if row["status"] != "runnable":
                write(attempts, dict(key, event="frozen_technical_missing", stage=row["stage"], reason=row["reason"], at=stamp()))
                continue
            p = prompts[row["prompt_line"] - 1]
            write(attempts, dict(key, event="started", at=stamp()))
            try:
                formatted = apply_chat_template(processor, config, p["prompt"], num_images=6)
                images = [str(args.frames_root / f) for f in row["frames"]]
                result = generate(model, processor, formatted, image=images, temperature=frozen.TEMPERATURE,
                                  max_tokens=frozen.MAX_TOKENS, verbose=False)
            except Exception as err:
                write(attempts, dict(key, event="failed", stage="inference", reason=repr(err), at=stamp()))
                continue
            write(replies, dict(key, prompt_sha256=row["prompt_sha256"], frames_sha256=row["frames_sha256"],
                                raw_response=result.text, prompt_tokens=result.prompt_tokens,
                                generated_tokens=result.generation_tokens, original_prompt_tokens=original[row["qid"]]["prompt_tokens"]))
            write(attempts, dict(key, event="generated", at=stamp()))
            print(key["qid"], "recorded", flush=True)
        write(attempts, dict(event="run_finished", at=stamp()))


if __name__ == "__main__":
    main()
