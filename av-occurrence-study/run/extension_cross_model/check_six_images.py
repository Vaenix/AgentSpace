#!/usr/bin/env python3
"""Six-image reception check on synthetic, non-research images (protocol 2fac798).

Writes six 448 x 448 plain-colour PNGs, then runs the locked model, template,
image processing and decoding on them once. Structural pass: six image blocks,
486 image tokens and a generation prompt-token count equal to the processor's
own count. The colour list in the reply is recorded as a behavioural note.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import lock

COLOURS = (("red", (220, 30, 30)), ("green", (30, 160, 60)), ("blue", (30, 60, 220)),
           ("yellow", (240, 220, 30)), ("black", (0, 0, 0)), ("white", (255, 255, 255)))
PROMPT = ("Six images are supplied. Name the single main colour of each image, in the supplied order, "
          "as a numbered list from 1 to 6.")


def write_images(directory):
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, (name, rgb) in enumerate(COLOURS, 1):
        path = directory / f"{index}-{name}.png"
        Image.new("RGB", (448, 448), rgb).save(path)
        paths.append(path)
    return paths


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", type=Path, required=True, help="Local checkpoint of revision be92141c")
    cli.add_argument("--images-dir", type=Path, required=True, help="Directory for the six synthetic PNGs")
    cli.add_argument("--output", type=Path, required=True, help="Write a new JSON record")
    args = cli.parse_args()
    lock_data = lock.load_lock()
    checkpoint = args.model.resolve(strict=True)
    packages = lock.check_packages(lock_data)
    checkpoint_sha256 = lock.check_checkpoint(checkpoint, lock_data)
    paths = write_images(args.images_dir)

    import mlx.core as mx
    from mlx_vlm import generate
    started = perf_counter()
    runtime = lock.load_model(checkpoint, lock_data)
    load_s = round(perf_counter() - started, 3)
    prepared = lock.prepare(runtime, PROMPT, [str(path) for path in paths])
    mx.reset_peak_memory()
    started = perf_counter()
    generated = generate(runtime["model"], runtime["processor"], prepared["formatted"], image=[str(p) for p in paths],
                         temperature=lock_data["decoding"]["temperature"],
                         max_tokens=lock_data["decoding"]["max_tokens"], verbose=False)
    elapsed = round(perf_counter() - started, 3)
    reply = generated.text
    named = [name for name, _ in COLOURS]
    lowered = reply.lower()
    positions = [lowered.find(name) for name in named]
    structural = {
        "image_blocks": prepared["pixel_shape"][1],
        "image_tokens": prepared["image_tokens"],
        "pixel_shape": prepared["pixel_shape"],
        "expected_prompt_tokens": prepared["expected_prompt_tokens"],
        "generation_prompt_tokens": generated.prompt_tokens,
    }
    structural["passed"] = (structural["image_blocks"] == lock.NUM_IMAGES
                            and structural["image_tokens"] == lock.NUM_IMAGES * lock.IMAGE_TOKENS_PER_FRAME
                            and structural["expected_prompt_tokens"] == structural["generation_prompt_tokens"])
    record = {
        "protocol_commit": lock.PROTOCOL_COMMIT,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "images": [{"file": path.name, "colour": name, "rgb": list(rgb), "size": [448, 448],
                    "sha256": lock.frozen.file_sha256(path)} for path, (name, rgb) in zip(paths, COLOURS)],
        "prompt": PROMPT,
        "formatted_prompt_sha256": prepared["formatted_sha256"],
        "structural": structural,
        "reply": reply,
        "behavioural_note": {
            "all_six_colours_named": all(pos >= 0 for pos in positions),
            "named_in_supplied_order": all(pos >= 0 for pos in positions) and positions == sorted(positions),
        },
        "generated_tokens": generated.generation_tokens,
        "hit_token_budget": generated.generation_tokens >= lock_data["decoding"]["max_tokens"],
        "latency_s": elapsed, "load_s": load_s, "peak_memory_gb": generated.peak_memory,
        "parameter_dtypes": runtime["parameter_dtypes"],
        "device": str(mx.default_device()), "packages": packages, "checkpoint_sha256": checkpoint_sha256,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"structural": structural, "reply": reply}, ensure_ascii=False, indent=2))
    if not structural["passed"]:
        raise SystemExit("Structural six-image check failed; do not run the formal questions")


if __name__ == "__main__":
    main()
