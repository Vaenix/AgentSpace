#!/usr/bin/env python3
"""Follow-up diagnostics for the six-image check, on the same synthetic colour fields.

Added after the reception check passed structurally but named the colours
wrongly. Uses the locked model and settings. Separates delivery from
behaviour: (1) each image's 81 input features inside the six-image call are
compared with the same image sent alone; (2) each image is asked about alone;
(3) two, three and six images are asked about together. No research input is used.
"""
import argparse
import json
from pathlib import Path

import lock
from check_six_images import COLOURS, write_images

QUESTIONS = (
    ("single", None, "What is the colour of this image? Answer with one word."),
    ("two", (1, 3), "What colour is the first image, and what colour is the second image?"),
    ("three", (4, 5, 2), "Name the colour of image 1, image 2 and image 3."),
    ("six", (1, 2, 3, 4, 5, 6), "What colour is the third image? Answer with one word."),
    ("six", (1, 2, 3, 4, 5, 6), "What colour is the fifth image? Answer with one word."),
)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--images-dir", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    lock_data = lock.load_lock()
    checkpoint = args.model.resolve(strict=True)
    lock.check_packages(lock_data)
    lock.check_checkpoint(checkpoint, lock_data)
    paths = [str(path) for path in write_images(args.images_dir)]

    import mlx.core as mx
    from PIL import Image
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    runtime = lock.load_model(checkpoint, lock_data)
    model, processor = runtime["model"], runtime["processor"]

    def image_features(selected):
        formatted = apply_chat_template(processor, runtime["config"], "x", num_images=len(selected))
        inputs = processor(images=[Image.open(path).convert("RGB") for path in selected], text=formatted)
        embeds = model.get_input_embeddings(inputs["input_ids"], inputs["pixel_values"].astype(mx.bfloat16),
                                            pixel_attention_mask=inputs.get("pixel_attention_mask")).inputs_embeds
        positions = [i for i, flag in enumerate((inputs["input_ids"][0] == processor.image_token_id).tolist()) if flag]
        return embeds[0][mx.array(positions)].astype(mx.float32)

    def ask(selected, question):
        formatted = apply_chat_template(processor, runtime["config"], question, num_images=len(selected))
        return generate(model, processor, formatted, image=selected, temperature=0.0, max_tokens=40,
                        verbose=False).text

    six = image_features(paths)
    per = lock.IMAGE_TOKENS_PER_FRAME
    placement = []
    for index, path in enumerate(paths):
        alone = image_features([path])
        same = mx.abs(six[per * index:per * (index + 1)] - alone).max().item()
        others = min(mx.abs(six[per * other:per * (other + 1)] - alone).max().item()
                     for other in range(len(paths)) if other != index)
        placement.append({"image": index + 1, "colour": COLOURS[index][0],
                          "max_abs_diff_to_same_image_alone": same, "min_max_abs_diff_to_other_images": others})
    answers = []
    for kind, order, question in QUESTIONS:
        groups = [(n,) for n in range(1, 7)] if order is None else [order]
        for group in groups:
            answers.append({"kind": kind, "images": [COLOURS[n - 1][0] for n in group], "question": question,
                            "reply": ask([paths[n - 1] for n in group], question)})
    record = {"protocol_commit": lock.PROTOCOL_COMMIT, "decoding": "temperature 0, max 40 tokens",
              "feature_placement": placement, "answers": answers}
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
