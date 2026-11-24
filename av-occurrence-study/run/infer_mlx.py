#!/usr/bin/env python3
"""Run fixed-budget event-occurrence prompts with a local Qwen2.5-VL checkpoint."""
import argparse
from decimal import Decimal
import json
import math
import re
import sys
import time
from importlib.metadata import version
from pathlib import Path


MODEL_ID = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
CONDITIONS = ("visual", "flat", "paired", "blind")
# strict: the event must be exactly one displayed evidence code (the original rule).
# matched_seconds: also accept "<code> <start>–<end>s" when both printed bounds equal
# the bounds this prompt displayed for that code.
PARSE_RULES = ("strict", "matched_seconds")
SECONDS = r"(\d+(?:\.\d+)?)"
CODE_WITH_SECONDS = re.compile(rf"(\S+)\s+{SECONDS}\s*s?\s*[–-]\s*{SECONDS}\s*s?")
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


def seconds_label(value):
    """Seconds exactly as the prompt prints them."""
    return f"{value:g}"


def interval_label(interval):
    return f"{seconds_label(interval[0])}–{seconds_label(interval[1])}s"


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
        label = interval_label(interval)
        frame_labels = []
        for frame in segment["frames"]:
            images.append(frame["path"])
            frame_labels.append(f"image {len(images)} at {frame['time_s']:g}s")
        visual = f"{code} {label} | " + ", ".join(frame_labels)
        sound = segment["sound_description"].strip() or "[unavailable]"
        visual_lines.append(visual)
        sound_line = f"{code} {label} | sound: {sound}"
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


def parse_answer(answer, option_ids):
    """Return (stable answer ID, field status); status is ok, null or invalid."""
    if answer is None:
        return None, "null"
    if isinstance(answer, str) and answer in option_ids:
        return option_ids[answer], "ok"
    return None, "invalid"


def parse_event(event, segment_intervals, rule):
    """Return (physical interval, field status); status is ok, null, invalid or conflict.

    Intervals always come from this prompt's own code mapping, so a reversed
    code mapping restores the window that the prompt displayed for that code.
    """
    if event is None:
        return None, "null"
    if not isinstance(event, str):
        return None, "invalid"
    text = event if rule == "strict" else event.strip()
    if text in segment_intervals:
        return segment_intervals[text], "ok"
    if rule == "strict":
        return None, "invalid"
    match = CODE_WITH_SECONDS.fullmatch(text)
    if not match or match[1] not in segment_intervals:
        return None, "invalid"
    interval = segment_intervals[match[1]]
    printed = tuple(Decimal(value) for value in match.group(2, 3))
    displayed = tuple(Decimal(seconds_label(value)) for value in interval)
    return (interval, "ok") if printed == displayed else (None, "conflict")


def parse_reply(response, option_ids, segment_intervals, rule="strict"):
    """Decode one model reply under a parse rule.

    Returns answer_id, interval_s, result_format and the per-field statuses.
    Under strict, result_format keeps its original values (ok, invalid_json,
    invalid_keys, invalid_value; explicit nulls count as ok). Under
    matched_seconds it also separates abstained, partially_abstained and
    code_seconds_conflict.
    """
    if rule not in PARSE_RULES:
        raise ValueError(f"unknown parse rule: {rule}")
    parsed = {"answer_id": None, "interval_s": None, "answer_status": None, "event_status": None}
    text = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
    try:
        reply = json.loads(fenced[1] if fenced else text)
    except json.JSONDecodeError:
        return dict(parsed, result_format="invalid_json")
    if not isinstance(reply, dict) or set(reply) != {"answer", "event"}:
        return dict(parsed, result_format="invalid_keys")

    answer_id, answer_status = parse_answer(reply["answer"], option_ids)
    interval, event_status = parse_event(reply["event"], segment_intervals, rule)
    statuses = answer_status, event_status
    if rule == "strict":
        result_format = "invalid_value" if "invalid" in statuses else "ok"
    elif "conflict" in statuses:
        result_format = "code_seconds_conflict"
    elif "invalid" in statuses:
        result_format = "invalid_value"
    elif statuses == ("null", "null"):
        result_format = "abstained"
    elif "null" in statuses:
        result_format = "partially_abstained"
    else:
        result_format = "ok"
    return {
        "answer_id": answer_id, "interval_s": interval, "result_format": result_format,
        "answer_status": answer_status, "event_status": event_status,
    }


def decode(response, option_ids, segment_intervals, rule="strict"):
    parsed = parse_reply(response, option_ids, segment_intervals, rule)
    return parsed["answer_id"], parsed["interval_s"], parsed["result_format"]


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
            parsed = parse_reply(generation.text, option_ids, intervals, args.parse_rule)
            record = {
                "condition": condition,
                "pair_id": row["pair_id"],
                "qid": row["qid"],
                "answer_id": parsed["answer_id"],
                "interval_s": parsed["interval_s"],
                "result_format": parsed["result_format"],
                "parse_rule": args.parse_rule,
                "answer_status": parsed["answer_status"],
                "event_status": parsed["event_status"],
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
    parser.add_argument(
        "--parse-rule", choices=PARSE_RULES, default="strict",
        help="strict accepts only a bare evidence code; matched_seconds also accepts the code followed by "
             "the start–end seconds this prompt displayed for it (default: strict)",
    )
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
