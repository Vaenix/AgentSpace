#!/usr/bin/env python3
"""Generate the extension replies in calls.jsonl order, reusing every stored reply.

Before the model is loaded, the script checks the frozen text inputs, all 66
frame files and both checkpoint shards. Each runnable key is generated once,
without conversation history. A key that already has a complete raw reply in
replies.jsonl is reused and never regenerated. A failed key is recorded in
attempts.jsonl with its reason and may be retried by a later invocation on the
same inputs. The six frozen technical_missing keys are never run.
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

import frozen

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


def stored_replies(path, calls):
    by_key = {frozen.key_of(call): call for call in calls}
    replies = {}
    for row in read_ledger(path):
        key = frozen.key_of(row)
        call = by_key.get(key)
        if call is None or call["status"] != "runnable":
            raise ValueError(f"{path}: {key} is not a runnable frozen key")
        if key in replies:
            raise ValueError(f"{path}: duplicate reply for {key}")
        if any(row[name] != call[name] for name in LEDGER_IDENTITY) or row["prompt_sha256"] != call["prompt_sha256"]:
            raise ValueError(f"{path}: {key} does not match its frozen call")
        if not isinstance(row.get("raw_response"), str):
            raise ValueError(f"{path}: {key} has no raw reply")
        replies[key] = row
    return replies


def load_parser():
    parser_file = frozen.RUN / "infer_mlx.py"
    if frozen.file_sha256(parser_file) != frozen.PARSER_SHA256:
        raise ValueError("infer_mlx.py differs from the parser pinned by the protocol")
    sys.path.insert(0, str(frozen.RUN))
    from infer_mlx import decode
    return decode


def check_frames(frames_root, frame_sha256):
    for relative, expected in frame_sha256.items():
        if frozen.file_sha256(frames_root / relative) != expected:
            raise ValueError(f"Frame differs from its frozen fingerprint: {relative}")


def check_checkpoint(checkpoint):
    measured = {}
    for name, expected in frozen.SHARD_SHA256.items():
        measured[name] = frozen.file_sha256(checkpoint / name)
        if measured[name] != expected:
            raise ValueError(f"Checkpoint shard differs from revision {frozen.MODEL_REVISION[:8]}: {name}")
    others = sorted(path.name for path in checkpoint.iterdir() if path.is_file() and path.name not in measured)
    return dict(measured, **{name: frozen.file_sha256(checkpoint / name) for name in others})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", type=Path, required=True, help="Local MLX checkpoint of revision fdcc572e")
    cli.add_argument("--frames-root", type=Path, required=True,
                     help="Directory holding frames/<pair>/window-*-frame-*.png from the freeze step")
    cli.add_argument("--results-dir", type=Path, default=frozen.HERE,
                     help="Batch directory holding replies.jsonl and attempts.jsonl (default: this directory)")
    args = cli.parse_args()
    checkpoint = args.model.resolve(strict=True)
    frames_root = args.frames_root.resolve(strict=True)
    replies_file = args.results_dir / "replies.jsonl"
    attempts_file = args.results_dir / "attempts.jsonl"

    inputs = frozen.load()
    calls = inputs["calls"]
    decode = load_parser()
    check_frames(frames_root, inputs["frame_sha256"])
    replies = stored_replies(replies_file, calls)
    read_ledger(attempts_file)
    pending = [call for call in calls if call["status"] == "runnable" and frozen.key_of(call) not in replies]
    run_id = now()
    runner_sha256 = frozen.file_sha256(Path(__file__))
    counts = {"generated": 0, "reused": 0, "failed": 0, "frozen_technical_missing": 0}

    with attempts_file.open("a", encoding="utf-8") as attempts, replies_file.open("a", encoding="utf-8") as ledger:
        def log(event, call=None, **fields):
            row = {"run_id": run_id, "at": now(), "event": event}
            if call is not None:
                row.update({name: call[name] for name in LEDGER_IDENTITY})
            row.update(fields)
            attempts.write(json.dumps(row, ensure_ascii=False) + "\n")
            attempts.flush()

        log("run_started", pending=len(pending), stored=len(replies), runner_sha256=runner_sha256,
            input_sha256=inputs["fingerprints"], checkpoint=str(checkpoint), frames_root=str(frames_root),
            python=platform.python_version(), platform=platform.platform())
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

        for call in calls:
            key = frozen.key_of(call)
            if call["status"] == "technical_missing":
                counts["frozen_technical_missing"] += 1
                log("frozen_technical_missing", call, stage=call["stage"], reason=call["reason"])
                continue
            if key in replies:
                counts["reused"] += 1
                log("reused", call, generated_run_id=replies[key]["run_id"])
                continue
            prompt = call["prompt"]
            log("started", call)
            try:
                images = [str(frames_root / path) for path in prompt["frames"]]
                for path, expected in zip(images, call["frames_sha256"]):
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
                log("failed", call, error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(limit=4))
                print(call["run_order"], key, "FAILED", error, flush=True)
                continue
            answer, interval, result_format = decode(generated.text, call["option_codes"], call["segment_codes"])
            row = {name: call[name] for name in LEDGER_IDENTITY}
            row.update(
                prompt_sha256=call["prompt_sha256"], answer_id=answer,
                action=prompt["semantic_actions"].get(answer), interval_s=interval,
                physical_window=None if interval is None else prompt["physical_window_order"].index(interval) + 1,
                result_format=result_format, raw_response=generated.text,
                prompt_tokens=generated.prompt_tokens, generated_tokens=generated.generation_tokens,
                latency_s=elapsed, peak_memory_gb=generated.peak_memory, temperature=frozen.TEMPERATURE,
                max_tokens=frozen.MAX_TOKENS, model_revision=frozen.MODEL_REVISION,
                mlx_version=version("mlx"), mlx_vlm_version=version("mlx-vlm"), load_s=load_seconds,
                run_id=run_id, generated_at=now(),
            )
            ledger.write(json.dumps(row, ensure_ascii=False) + "\n")
            ledger.flush()
            replies[key] = row
            counts["generated"] += 1
            log("generated", call)
            print(call["run_order"], *key, row["action"], row["physical_window"], result_format, flush=True)
        log("run_finished", **counts)
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
