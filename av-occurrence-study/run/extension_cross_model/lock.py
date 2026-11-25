"""Shared checks for the SmolVLM2 cross-model control, frozen in commit 2fac798.

Text-only helpers (protocol fingerprints, the reference ledger, the native
chat template) need only the standard library and the archived files. The
model helpers import MLX lazily so offline replay and tests run without it.
"""
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RESULTS_1124 = HERE.parent / "participant_extension" / "results_1124"
sys.path.insert(0, str(RESULTS_1124))
import frozen  # noqa: E402  (extension inputs frozen at 039e834 / 9b528fc)

PROTOCOL_COMMIT = "2fac79828c805a77c42b76af49c7ae4491c3bdbf"
PROTOCOL_SHA256 = {
    "protocol.md": "e09236027deea47d3b01ed01d27321325ee49de63d0ee366ab39ef3b94a26403",
    "model_lock.json": "416eae5a7e7fef02f6972a797bbeb3f8f29020020cf073e29f6f6d7078c4ce14",
}
REFERENCE_COMMIT = "3df02a8e1b84d876ebdf8b54b9826d803860fcf1"
REFERENCE_SHA256 = {
    "replies.jsonl": "b87936a0fd0d2b0d5d1f75f93f2d31dd81ad7fbc942fe2491d12ed21adc86aaa",
    "attempts.jsonl": "0bd2342c5ae8b04f41e46193b61c010d3c6b286eb3a4a730a23836235bff422b",
}
REFERENCE_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit@fdcc572e"
NUM_IMAGES = 6
IMAGE_TOKENS_PER_FRAME = 81


def load_lock():
    for name, expected in PROTOCOL_SHA256.items():
        if frozen.file_sha256(HERE / name) != expected:
            raise ValueError(f"{name} differs from the protocol commit {PROTOCOL_COMMIT[:7]}")
    return json.loads((HERE / "model_lock.json").read_text(encoding="utf-8"))


def reference_paths():
    paths = {name: RESULTS_1124 / name for name in REFERENCE_SHA256}
    for name, path in paths.items():
        if frozen.file_sha256(path) != REFERENCE_SHA256[name]:
            raise ValueError(f"Qwen reference {name} differs from {REFERENCE_COMMIT[:7]}")
    return paths


def check_checkpoint(checkpoint, lock):
    """Verify every locked file of the checkpoint; return the measured fingerprints."""
    measured = {}
    for name, expected in lock["model"]["file_sha256"].items():
        measured[name] = frozen.file_sha256(Path(checkpoint) / name)
        if measured[name] != expected:
            raise ValueError(f"Checkpoint file differs from revision {lock['model']['revision'][:8]}: {name}")
    return measured


def check_packages(lock):
    from importlib.metadata import version
    import platform
    found = {name: version(name) for name in lock["runtime"]["packages"]}
    if found != lock["runtime"]["packages"] or platform.python_version() != lock["runtime"]["python"]:
        raise ValueError(f"Runtime differs from the lock: python {platform.python_version()}, {found}")
    return found


def native_messages(prompt_text, num_images=NUM_IMAGES):
    return [{"role": "user", "content": [{"type": "image"}] * num_images + [{"type": "text", "text": prompt_text}]}]


def render_native(template, prompt_text, num_images=NUM_IMAGES):
    """Render the stored chat template the way transformers does, with no library in between."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    return env.from_string(template).render(messages=native_messages(prompt_text, num_images),
                                            add_generation_prompt=True)


def stored_template(checkpoint, lock):
    template = json.loads((Path(checkpoint) / "chat_template.json").read_text(encoding="utf-8"))["chat_template"]
    if sha256(template.encode("utf-8")).hexdigest() != lock["chat_template"]["sha256_of_template_string"]:
        raise ValueError("chat_template.json differs from the lock")
    return template


def load_model(checkpoint, lock):
    """Load the locked checkpoint, cast to bfloat16 and disable image splitting; return the runtime pieces."""
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    model, processor = load(str(checkpoint), lazy=True)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    dtypes = sorted({str(value.dtype) for _, value in _leaves(model.parameters())})
    if any("float" in name and name != "mlx.core.bfloat16" for name in dtypes):
        raise ValueError(f"Floating parameters not all bfloat16: {dtypes}")
    image_processor = processor.image_processor
    stored = json.loads((Path(checkpoint) / "preprocessor_config.json").read_text(encoding="utf-8"))
    for name in ("do_convert_rgb", "do_resize", "resample", "do_rescale", "rescale_factor", "do_normalize",
                 "image_mean", "image_std", "do_pad", "max_image_size", "do_image_splitting"):
        value = getattr(image_processor, name)
        value = dict(value) if isinstance(value, dict) else list(value) if isinstance(value, tuple) else value
        if value != stored[name]:
            raise ValueError(f"Image processor {name}={value!r} differs from preprocessor_config.json")
    image_processor.do_image_splitting = lock["image_processing"]["override"]["do_image_splitting"]
    template = stored_template(checkpoint, lock)
    if processor.tokenizer.chat_template != template:
        raise ValueError("Loaded tokenizer chat template differs from chat_template.json")
    return {"model": model, "processor": processor, "config": load_config(str(checkpoint)),
            "template": template, "parameter_dtypes": dtypes}


def _leaves(tree, prefix=""):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _leaves(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(tree, (list, tuple)):
        for index, value in enumerate(tree):
            yield from _leaves(value, f"{prefix}.{index}")
    else:
        yield prefix, tree


def prepare(runtime, prompt_text, image_paths):
    """Format one call with the native template and count its tokens with the locked processor."""
    from mlx_vlm.prompt_utils import apply_chat_template
    from PIL import Image

    if len(image_paths) != NUM_IMAGES:
        raise ValueError("Each call needs six images")
    formatted = apply_chat_template(runtime["processor"], runtime["config"], prompt_text, num_images=NUM_IMAGES)
    if formatted != render_native(runtime["template"], prompt_text):
        raise ValueError("Formatted prompt differs from the native chat template rendering")
    processor = runtime["processor"]
    images = [Image.open(path).convert("RGB") for path in image_paths]
    inputs = processor(images=images, text=formatted)
    ids = inputs["input_ids"]
    image_tokens = int((ids == processor.image_token_id).sum().item())
    shape = tuple(int(n) for n in inputs["pixel_values"].shape)
    if image_tokens != NUM_IMAGES * IMAGE_TOKENS_PER_FRAME or shape[:2] != (1, NUM_IMAGES):
        raise ValueError(f"Processed input has {image_tokens} image tokens and pixel shape {shape}")
    return {"formatted": formatted, "formatted_sha256": sha256(formatted.encode("utf-8")).hexdigest(),
            "expected_prompt_tokens": int(ids.shape[1]), "image_tokens": image_tokens, "pixel_shape": list(shape)}
