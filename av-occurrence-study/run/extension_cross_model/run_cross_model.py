#!/usr/bin/env python3
"""Generate SmolVLM2 replies for the 72 extension call keys under protocol 2fac798.

Before the model is loaded the script checks the protocol files, the frozen
extension inputs, all 66 frames, the runtime packages and every checkpoint
file. Each runnable key is generated once with a fresh context. A key with a
stored reply in replies.jsonl is reused and never regenerated. A failed key is
logged in attempts.jsonl with its reason and may be retried later on the same
inputs. The six frozen technical_missing keys are never run.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import traceback
from time import perf_counter

import lock

frozen = lock.frozen
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


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", type=Path, required=True, help="Local checkpoint of revision be92141c")
    cli.add_argument("--frames-root", type=Path, required=True,
                     help="Directory holding frames/<pair>/window-*-frame-*.png from the freeze step")
    cli.add_argument("--results-dir", type=Path, default=lock.HERE,
                     help="Directory holding replies.jsonl and attempts.jsonl (default: this directory)")
    args = cli.parse_args()
    checkpoint = args.model.resolve(strict=True)
    frames_root = args.frames_root.resolve(strict=True)
    replies_file = args.results_dir / "replies.jsonl"
    attempts_file = args.results_dir / "attempts.jsonl"

    lock_data = lock.load_lock()
    decoding = lock_data["decoding"]
    inputs = frozen.load()
    calls = inputs["calls"]
    decode = load_parser()
    check_frames(frames_root, inputs["frame_sha256"])
    replies = stored_replies(replies_file, calls)
    read_ledger(attempts_file)
    pending = [call for call in calls if call["status"] == "runnable" and frozen.key_of(call) not in replies]
    run_id = now()
    runner_sha256 = frozen.file_sha256(Path(__file__))
    counts = {"generated": 0, "reused": 0, "failed": 0, "frozen_technical_missing": 0, "prompt_token_mismatch": 0}

    with attempts_file.open("a", encoding="utf-8") as attempts, replies_file.open("a", encoding="utf-8") as ledger:
        def log(event, call=None, **fields):
            row = {"run_id": run_id, "at": now(), "event": event}
            if call is not None:
                row.update({name: call[name] for name in LEDGER_IDENTITY})
            row.update(fields)
            attempts.write(json.dumps(row, ensure_ascii=False) + "\n")
            attempts.flush()

        log("run_started", pending=len(pending), stored=len(replies), protocol_commit=lock.PROTOCOL_COMMIT,
            runner_sha256=runner_sha256, lock_sha256=frozen.file_sha256(lock.__file__),
            input_sha256=inputs["fingerprints"], checkpoint=str(checkpoint), frames_root=str(frames_root),
            python=platform.python_version(), platform=platform.platform())
        runtime = load_s = packages = None
        if pending:
            packages = lock.check_packages(lock_data)
            log("checkpoint_verified", revision=lock_data["model"]["revision"],
                file_sha256=lock.check_checkpoint(checkpoint, lock_data), packages=packages)
            import mlx.core as mx
            from mlx_vlm import generate

            started = perf_counter()
            runtime = lock.load_model(checkpoint, lock_data)
            load_s = round(perf_counter() - started, 3)
            log("model_loaded", load_s=load_s, device=str(mx.default_device()),
                parameter_dtypes=runtime["parameter_dtypes"],
                do_image_splitting=runtime["processor"].image_processor.do_image_splitting)
            print(f"Model loaded in {load_s}s; {len(pending)} keys to generate", flush=True)

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
                prepared = lock.prepare(runtime, prompt["prompt"], images)
                mx.reset_peak_memory()
                started = perf_counter()
                generated = generate(runtime["model"], runtime["processor"], prepared["formatted"], image=images,
                                     temperature=decoding["temperature"], max_tokens=decoding["max_tokens"],
                                     verbose=False)
                elapsed = round(perf_counter() - started, 3)
                if not isinstance(generated.text, str):
                    raise TypeError("Generation returned no text")
            except Exception as error:
                counts["failed"] += 1
                log("failed", call, error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(limit=4))
                print(call["run_order"], key, "FAILED", error, flush=True)
                continue
            mismatch = generated.prompt_tokens != prepared["expected_prompt_tokens"]
            counts["prompt_token_mismatch"] += mismatch
            answer, interval, result_format = decode(generated.text, call["option_codes"], call["segment_codes"])
            row = {name: call[name] for name in LEDGER_IDENTITY}
            row.update(
                prompt_sha256=call["prompt_sha256"], answer_id=answer,
                action=prompt["semantic_actions"].get(answer), interval_s=interval,
                physical_window=None if interval is None else prompt["physical_window_order"].index(interval) + 1,
                result_format=result_format, raw_response=generated.text,
                formatted_prompt_sha256=prepared["formatted_sha256"],
                prompt_tokens=generated.prompt_tokens, expected_prompt_tokens=prepared["expected_prompt_tokens"],
                prompt_token_mismatch=mismatch, image_tokens=prepared["image_tokens"],
                generated_tokens=generated.generation_tokens,
                hit_token_budget=generated.generation_tokens >= decoding["max_tokens"],
                latency_s=elapsed, peak_memory_gb=generated.peak_memory,
                temperature=decoding["temperature"], max_tokens=decoding["max_tokens"],
                model=lock_data["model"]["repository"], model_revision=lock_data["model"]["revision"],
                dtype="bfloat16", do_image_splitting=False, protocol_commit=lock.PROTOCOL_COMMIT,
                mlx_version=packages["mlx"], mlx_vlm_version=packages["mlx-vlm"], load_s=load_s,
                run_id=run_id, generated_at=now(),
            )
            ledger.write(json.dumps(row, ensure_ascii=False) + "\n")
            ledger.flush()
            replies[key] = row
            counts["generated"] += 1
            log("generated", call)
            print(call["run_order"], *key, row["action"], row["physical_window"], result_format,
                  generated.prompt_tokens, prepared["expected_prompt_tokens"], flush=True)
        log("run_finished", **counts)
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
