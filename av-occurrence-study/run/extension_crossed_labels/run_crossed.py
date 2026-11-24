#!/usr/bin/env python3
"""Generate the joint-reversal replies for the 22 runnable keys of protocol af01c3d.

Before the model is loaded the script checks the frozen protocol files, the
extension's frozen inputs and archived three-condition ledgers (3df02a8), both
construction paths of every joint prompt, the pinned parser, the 66 original
448 x 448 scene images and both checkpoint shards. Each runnable key loads its
six original images in the archived order. The call is independent, without
conversation history, at temperature 0 with a 64-token cap, using the same
template and image processor call as the intact run. Only the first returned
text of a key is kept: a key that already has a reply in replies.jsonl is never
regenerated. A failed key is logged in attempts.jsonl with its reason and
retried by a later invocation on identical inputs. The two frozen P11_104
technical misses are never run. The three archived conditions are never
regenerated; this script writes only its own ledgers.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import traceback
from time import perf_counter

import crossed_inputs
from crossed_inputs import frozen

LEDGER_IDENTITY = ("run_order", "video_id", "pair_id", "qid", "ordinal", "condition")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def read_ledger(path):
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError(f"{path} ends with an unterminated row; inspect it before continuing")
    return frozen.read_jsonl(data)


def stored_replies(path, keys):
    by_key = {frozen.key_of(key): key for key in keys}
    replies = {}
    for row in read_ledger(path):
        identity = frozen.key_of(row)
        key = by_key.get(identity)
        if key is None or key["status"] != "runnable":
            raise ValueError(f"{path}: {identity} is not a runnable planned key")
        if identity in replies:
            raise ValueError(f"{path}: duplicate reply for {identity}")
        if any(row[name] != key[name] for name in LEDGER_IDENTITY) or row["prompt_sha256"] != key["prompt_sha256"]:
            raise ValueError(f"{path}: {identity} does not match its planned key")
        if row.get("frames_sha256") != key["frames_sha256"] or not isinstance(row.get("raw_response"), str):
            raise ValueError(f"{path}: {identity} lacks the planned images or raw text")
        replies[identity] = row
    return replies


def archived_prompt_tokens(archived_replies):
    tokens = {}
    for row in archived_replies:
        tokens.setdefault(row["qid"], {})[row["condition"]] = row["prompt_tokens"]
    return tokens


def load_parser():
    parser_file = frozen.RUN / "infer_mlx.py"
    if frozen.file_sha256(parser_file) != frozen.PARSER_SHA256:
        raise ValueError("infer_mlx.py differs from the parser pinned by the protocol")
    sys.path.insert(0, str(frozen.RUN))
    from infer_mlx import decode
    return decode


def check_checkpoint(checkpoint):
    measured = {}
    for name, expected in frozen.SHARD_SHA256.items():
        measured[name] = frozen.file_sha256(checkpoint / name)
        if measured[name] != expected:
            raise ValueError(f"Checkpoint shard differs from revision {frozen.MODEL_REVISION[:8]}: {name}")
    others = sorted(path.name for path in checkpoint.iterdir() if path.is_file() and path.name not in measured)
    return dict(measured, **{name: frozen.file_sha256(checkpoint / name) for name in others})


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--model", type=Path, required=True, help="Local MLX checkpoint of revision fdcc572e")
    cli.add_argument("--frames-root", type=Path, required=True,
                     help="Directory holding frames/<pair>/window-*-frame-*.png from the extension freeze step")
    cli.add_argument("--results-dir", type=Path, default=crossed_inputs.HERE,
                     help="Directory holding replies.jsonl and attempts.jsonl (default: this directory)")
    args = cli.parse_args()
    checkpoint = args.model.resolve(strict=True)
    frames_root = args.frames_root.resolve(strict=True)
    replies_file = args.results_dir / "replies.jsonl"
    attempts_file = args.results_dir / "attempts.jsonl"

    inputs = crossed_inputs.load()
    keys = inputs["keys"]
    reference_tokens = archived_prompt_tokens(inputs["archived_replies"])
    decode = load_parser()
    frame_sha256 = crossed_inputs.check_frames(frames_root, keys)
    replies = stored_replies(replies_file, keys)
    read_ledger(attempts_file)
    pending = [key for key in keys if key["status"] == "runnable" and frozen.key_of(key) not in replies]
    run_id = now()
    counts = {"generated": 0, "reused": 0, "failed": 0, "frozen_technical_missing": 0}

    with attempts_file.open("a", encoding="utf-8") as attempts, replies_file.open("a", encoding="utf-8") as ledger:
        def log(event, key=None, **fields):
            row = {"run_id": run_id, "at": now(), "event": event}
            if key is not None:
                row.update({name: key[name] for name in LEDGER_IDENTITY})
            row.update(fields)
            attempts.write(json.dumps(row, ensure_ascii=False) + "\n")
            attempts.flush()

        log("run_started", pending=len(pending), stored=len(replies),
            runner_sha256=frozen.file_sha256(Path(__file__)),
            inputs_sha256=frozen.file_sha256(Path(crossed_inputs.__file__)),
            input_sha256=inputs["fingerprints"], checkpoint=str(checkpoint), frames_root=str(frames_root),
            python=platform.python_version(), platform=platform.platform())
        log("inputs_verified", runnable=sum(key["status"] == "runnable" for key in keys),
            construction_paths_identical=True, planned_rows_identical=True,
            prompt_sha256={key["qid"]: key["prompt_sha256"] for key in keys if key["status"] == "runnable"})
        log("frames_verified", count=len(frame_sha256), size=list(crossed_inputs.IMAGE_SIZE),
            frame_sha256=frame_sha256)
        model = processor = config = load_seconds = None
        if pending:
            checkpoint_sha256 = check_checkpoint(checkpoint)
            log("checkpoint_verified", revision=frozen.MODEL_REVISION, file_sha256=checkpoint_sha256)
            import mlx.core as mx
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config

            started = perf_counter()
            model, processor = load(str(checkpoint))
            config = load_config(str(checkpoint))
            load_seconds = round(perf_counter() - started, 3)
            log("model_loaded", load_s=load_seconds, device=str(mx.default_device()),
                mlx_version=version("mlx"), mlx_vlm_version=version("mlx-vlm"))
            print(f"Model loaded in {load_seconds}s; {len(pending)} keys to generate", flush=True)

        for key in keys:
            identity = frozen.key_of(key)
            if key["status"] == "technical_missing":
                counts["frozen_technical_missing"] += 1
                log("frozen_technical_missing", key, stage=key["stage"], reason=key["reason"])
                continue
            if identity in replies:
                counts["reused"] += 1
                log("reused", key, generated_run_id=replies[identity]["run_id"])
                continue
            prompt = key["prompt"]
            log("started", key)
            try:
                images = [str(frames_root / path) for path in prompt["frames"]]
                for path, expected in zip(images, key["frames_sha256"]):
                    if frozen.file_sha256(path) != expected:
                        raise ValueError(f"Frame changed during the run: {path}")
                formatted = apply_chat_template(processor, config, prompt["prompt"], num_images=len(images))
                mx.reset_peak_memory()
                started = perf_counter()
                generated = generate(model, processor, formatted, image=images,
                                     temperature=frozen.TEMPERATURE, max_tokens=frozen.MAX_TOKENS, verbose=False)
                elapsed = round(perf_counter() - started, 3)
                if not isinstance(generated.text, str):
                    raise TypeError("Generation returned no text")
            except Exception as error:
                counts["failed"] += 1
                log("failed", key, error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(limit=4))
                print(key["run_order"], identity, "FAILED", error, flush=True)
                continue
            answer, interval, result_format = decode(generated.text, key["option_codes"], key["segment_codes"])
            row = {name: key[name] for name in LEDGER_IDENTITY}
            row.update(
                prompt_sha256=key["prompt_sha256"], frames_sha256=key["frames_sha256"],
                answer_id=answer, action=prompt["semantic_actions"].get(answer), interval_s=interval,
                physical_window=None if interval is None else prompt["physical_window_order"].index(interval) + 1,
                result_format=result_format, raw_response=generated.text,
                prompt_tokens=generated.prompt_tokens,
                archived_prompt_tokens=reference_tokens.get(key["qid"], {}),
                generated_tokens=generated.generation_tokens,
                latency_s=elapsed, peak_memory_gb=generated.peak_memory, temperature=frozen.TEMPERATURE,
                max_tokens=frozen.MAX_TOKENS, model_revision=frozen.MODEL_REVISION,
                mlx_version=version("mlx"), mlx_vlm_version=version("mlx-vlm"), load_s=load_seconds,
                run_id=run_id, generated_at=now(),
            )
            ledger.write(json.dumps(row, ensure_ascii=False) + "\n")
            ledger.flush()
            replies[identity] = row
            counts["generated"] += 1
            log("generated", key, prompt_tokens=generated.prompt_tokens,
                generated_tokens=generated.generation_tokens)
            print(key["run_order"], *identity, row["action"], row["physical_window"], result_format,
                  generated.prompt_tokens, flush=True)
        log("run_finished", **counts)
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
