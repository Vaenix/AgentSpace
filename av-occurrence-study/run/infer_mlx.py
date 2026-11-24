#!/usr/bin/env python3
"""Run fixed-budget event-occurrence prompts with a local Qwen2.5-VL checkpoint."""
import argparse
import json
import math
import re
import sys
import time
from importlib.metadata import version
from pathlib import Path


MODEL_ID = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
CONDITIONS = ("visual", "flat", "paired", "blind")
QUESTION_KEYS = {"pair_id", "qid", "question", "options", "segments"}
SEGMENT_KEYS = {"interval_s", "sound_description", "frames"}
FRAME_KEYS = {"path", "time_s"}


def require_keys(value, keys, noun):
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{noun} must have exactly these keys: {', '.join(sorted(keys))}")


def valid_time(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def load_manifest(path):
    """Load only information that is allowed to be passed to the model."""
    seen = set()
    questions = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            require_keys(row, QUESTION_KEYS, "question")
            for key in ("pair_id", "qid", "question"):
                if not isinstance(row[key], str) or not row[key].strip():
                    raise ValueError(f"{key} must be a nonempty string")
            identity = row["pair_id"], row["qid"]
            if identity in seen:
                raise ValueError(f"duplicate question: {identity}")
            seen.add(identity)
            options = row["options"]
            if not isinstance(options, dict) or not 2 <= len(options) <= 10:
                raise ValueError("options must map 2–10 stable semantic IDs to descriptions")
            if not all(k.strip() and isinstance(v, str) and v.strip() for k, v in options.items()):
                raise ValueError("option IDs and descriptions must be nonempty")
            if len({v.strip() for v in options.values()}) != len(options):
                raise ValueError("option descriptions must be distinct")

            segments = row["segments"]
            if not isinstance(segments, list) or not segments:
                raise ValueError("segments must be a nonempty list")
            last_end = -1
            last_frame = -1
            for segment in segments:
                require_keys(segment, SEGMENT_KEYS, "segment")
                interval = segment["interval_s"]
                if (
                    not isinstance(interval, list) or len(interval) != 2
                    or not all(map(valid_time, interval))
                    or not last_end <= interval[0] < interval[1]
                ):
                    raise ValueError("segment intervals must be finite, sorted, and nonoverlapping")
                last_end = interval[1]
                if not isinstance(segment["sound_description"], str):
                    raise ValueError("sound_description must be a string")
                frames = segment["frames"]
                if not isinstance(frames, list) or not frames:
                    raise ValueError("each segment must contain at least one frame")
                for frame in frames:
                    require_keys(frame, FRAME_KEYS, "frame")
                    if not valid_time(frame["time_s"]) or frame["time_s"] <= last_frame:
                        raise ValueError("frame times must strictly increase across segments")
                    last_frame = frame["time_s"]
                    if not isinstance(frame["path"], str) or not frame["path"].strip():
                        raise ValueError("frame path must be a nonempty string")
                    frame_path = (path.parent / frame["path"]).resolve(strict=True)
                    if not frame_path.is_file() or frame_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                        raise ValueError("frame must be a local PNG or JPEG file")
                    frame["path"] = str(frame_path)
            questions.append(row)
        except (ValueError, OSError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    if not questions:
        raise ValueError("input manifest contains no questions")
    return questions


def prompt_for(row, condition):
    """Return the prompt, images and opaque-output lookups for one condition."""
    option_ids = {chr(ord("A") + index): key for index, key in enumerate(row["options"])}
    options = "\n".join(f"{code}. {row['options'][key]}" for code, key in option_ids.items())
    parts = [
        row["question"],
        f"Options:\n{options}",
        'Return only a JSON object with keys "answer" (option letter) and "event" (evidence event ID). '
        "Use null for either value if the available evidence does not support it.",
    ]
    images = []
    segment_intervals = {}
    if condition == "blind":
        return "\n\n".join(parts), images, option_ids, segment_intervals

    visual_lines, sound_lines, paired_lines = [], [], []
    for index, segment in enumerate(row["segments"], 1):
        code = f"E{index:02}"
        interval = segment["interval_s"]
        segment_intervals[code] = interval
        interval_label = f"{interval[0]:g}–{interval[1]:g}s"
        frame_labels = []
        for frame in segment["frames"]:
            images.append(frame["path"])
            frame_labels.append(f"image {len(images)} at {frame['time_s']:g}s")
        visual = f"{code} {interval_label} | " + ", ".join(frame_labels)
        sound = segment["sound_description"].strip() or "[unavailable]"
        visual_lines.append(visual)
        sound_line = f"{code} {interval_label} | sound: {sound}"
        sound_lines.append(sound_line)
        paired_lines.extend((visual, sound_line))

    if condition == "flat":
        evidence_lines = visual_lines + sound_lines
    elif condition == "paired":
        evidence_lines = paired_lines
    else:
        evidence_lines = visual_lines
    parts.append("Evidence entries (image numbers follow the supplied image order):\n" + "\n".join(evidence_lines))
    return "\n\n".join(parts), images, option_ids, segment_intervals


def decode(response, option_ids, segment_intervals):
    text = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
    try:
        reply = json.loads(fenced[1] if fenced else text)
    except json.JSONDecodeError:
        return None, None, "invalid_json"
    if not isinstance(reply, dict) or set(reply) != {"answer", "event"}:
        return None, None, "invalid_keys"

    answer, event = reply["answer"], reply["event"]
    answer_valid = answer is None or isinstance(answer, str) and answer in option_ids
    event_valid = event is None or isinstance(event, str) and event in segment_intervals
    return (
        option_ids.get(answer) if isinstance(answer, str) else None,
        segment_intervals.get(event) if isinstance(event, str) else None,
        "ok" if answer_valid and event_valid else "invalid_value",
    )


def run(args, stream):
    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    checkpoint = str(args.model.resolve(strict=True))
    questions = load_manifest(args.manifest)
    start = time.perf_counter()
    model, processor = load(checkpoint)
    config = load_config(checkpoint)
    load_seconds = round(time.perf_counter() - start, 3)
    print(f"Loaded local checkpoint in {load_seconds:g}s ({len(questions)} questions)", file=sys.stderr)

    for row in questions:
        for condition in args.condition:
            prompt, images, option_ids, intervals = prompt_for(row, condition)
            formatted = apply_chat_template(processor, config, prompt, num_images=len(images))
            mx.reset_peak_memory()
            start = time.perf_counter()
            generation = generate(
                model, processor, formatted, image=images or None,
                max_tokens=args.max_tokens, temperature=0.0, verbose=False,
            )
            elapsed = round(time.perf_counter() - start, 3)
            answer, interval, result_format = decode(generation.text, option_ids, intervals)
            record = {
                "condition": condition,
                "pair_id": row["pair_id"],
                "qid": row["qid"],
                "answer_id": answer,
                "interval_s": interval,
                "result_format": result_format,
                "raw_response": generation.text,
                "latency_s": elapsed,
                "frame_count": len(images),
                "prompt_tokens": generation.prompt_tokens,
                "generation_tokens": generation.generation_tokens,
                "peak_memory_gb": generation.peak_memory,
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "checkpoint_path": checkpoint,
                "model_load_s": load_seconds,
                "mlx_version": version("mlx"),
                "mlx_vlm_version": version("mlx-vlm"),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Gold-free JSONL with local candidate frames")
    parser.add_argument("--model", type=Path, required=True, help=f"Local MLX snapshot for {MODEL_ID}")
    parser.add_argument("--condition", choices=CONDITIONS, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Write predictions to a new JSONL file")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if len(set(args.condition)) != len(args.condition):
        parser.error("--condition cannot be repeated with the same value")
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            run(args, stream)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
