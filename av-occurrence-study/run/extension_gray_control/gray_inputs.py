"""Load and verify the gray-image control inputs frozen in commit 4e87d1e.

Reads text and one PNG only; needs Git, Python 3.11+ and no media, GPU or model.
Every protocol file is compared with its SHA-256 at the freeze commit. The
runnable keys take their exact prompt text and mappings from the extension's
frozen inputs (results_1124/frozen.py), and the paired intact reference is the
archived original-condition reply ledger of PR #11.
"""
import csv
from hashlib import sha256
import io
from pathlib import Path
import struct
import sys
import zlib

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
EXTENSION_RESULTS = RUN / "participant_extension" / "results_1124"
sys.path.insert(0, str(EXTENSION_RESULTS))

import frozen  # noqa: E402  (extension inputs frozen in 039e834)

PROTOCOL_COMMIT = "4e87d1e0016161e1c0c09c1faaeddcc9d48e2e66"
PROTOCOL_PLAN_COMMIT = "a7df2e2c4e5c8ac8b0ca820f1e0e8d59c5550206"
INTACT_RESULTS_COMMIT = "3df02a8e1b84d876ebdf8b54b9826d803860fcf1"
PROTOCOL_SHA256 = {
    "protocol.md": "cf6fa05a06a6e3a024cea4af990ada910dbe87cc56c5743f8f5d4850df6752fd",
    "planned.csv": "5e3de7c2c63962cd3533c8605cc1a6a4913fb634c34cd51af0cf488d252dffd2",
    "gray.png": "beae1a76505cf5bc29fbf1e519ddb554ff8624763cb9a8763d4c56c9a54215b2",
}
INTACT_REPLIES_SHA256 = "b87936a0fd0d2b0d5d1f75f93f2d31dd81ad7fbc942fe2491d12ed21adc86aaa"
INTACT_ATTEMPTS_SHA256 = "0bd2342c5ae8b04f41e46193b61c010d3c6b286eb3a4a730a23836235bff422b"
CONDITION = "original"
GRAY_SIZE = (448, 448)
GRAY_PIXEL = (128, 128, 128)
SLOTS = 6
PLANNED_KEYS = 24
RUNNABLE_KEYS = 22
GRAY_PNG = HERE / "gray.png"


def decode_png_rgb8(data):
    """Return (width, height, rows) for a non-interlaced 8-bit RGB PNG using only the standard library."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a PNG file")
    offset, header, idat = 8, None, b""
    while offset < len(data):
        length, kind = struct.unpack(">I4s", data[offset:offset + 8])
        body = data[offset + 8:offset + 8 + length]
        if zlib.crc32(kind + body) != struct.unpack(">I", data[offset + 8 + length:offset + 12 + length])[0]:
            raise ValueError(f"PNG chunk {kind!r} fails its CRC")
        offset += 12 + length
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
    width, height, depth, colour, _, _, interlace = header
    if (depth, colour, interlace) != (8, 2, 0):
        raise ValueError("Expected a non-interlaced 8-bit RGB PNG")
    raw, stride, rows, previous = zlib.decompress(idat), width * 3, [], bytearray(width * 3)
    for y in range(height):
        start = y * (stride + 1)
        kind, line = raw[start], bytearray(raw[start + 1:start + 1 + stride])
        for x in range(stride):
            left = line[x - 3] if x >= 3 else 0
            up = previous[x]
            corner = previous[x - 3] if x >= 3 else 0
            if kind == 1:
                line[x] = (line[x] + left) & 255
            elif kind == 2:
                line[x] = (line[x] + up) & 255
            elif kind == 3:
                line[x] = (line[x] + (left + up) // 2) & 255
            elif kind == 4:
                guess = left + up - corner
                pa, pb, pc = abs(guess - left), abs(guess - up), abs(guess - corner)
                line[x] = (line[x] + (left if pa <= pb and pa <= pc else up if pb <= pc else corner)) & 255
            elif kind != 0:
                raise ValueError(f"Unknown PNG filter {kind}")
        rows.append(bytes(line))
        previous = line
    return width, height, rows


def check_gray(data):
    """The shared slot image must be a 448 x 448 RGB image whose every pixel is (128,128,128)."""
    width, height, rows = decode_png_rgb8(data)
    if (width, height) != GRAY_SIZE:
        raise ValueError(f"gray.png is {width} x {height}, expected {GRAY_SIZE[0]} x {GRAY_SIZE[1]}")
    expected = bytes(GRAY_PIXEL) * width
    if any(row != expected for row in rows):
        raise ValueError("gray.png is not uniformly RGB(128,128,128)")
    return {"width": width, "height": height, "mode": "RGB", "bit_depth": 8, "pixel": list(GRAY_PIXEL),
            "pixels_checked": width * height}


def load():
    """Return the 24 planned keys in run order, the intact reference and every fingerprint used."""
    raw = {}
    for name, expected in PROTOCOL_SHA256.items():
        data = (HERE / name).read_bytes()
        if sha256(data).hexdigest() != expected:
            raise ValueError(f"{name} differs from the frozen protocol commit {PROTOCOL_COMMIT[:7]}")
        raw[name] = data
    gray = check_gray(raw["gray.png"])
    extension = frozen.load()
    calls = {frozen.key_of(call): call for call in extension["calls"]}
    planned = list(csv.DictReader(io.StringIO(raw["planned.csv"].decode("utf-8"))))
    originals = [call for call in extension["calls"] if call["condition"] == CONDITION]
    if len(planned) != PLANNED_KEYS or len(originals) != PLANNED_KEYS:
        raise ValueError("Expected 24 planned original-condition keys")

    keys = []
    for row, call in zip(planned, originals):
        if (int(row["run_order"]), row["video_id"], row["qid"], int(row["ordinal"])) != (
                call["run_order"], call["video_id"], call["qid"], call["ordinal"]):
            raise ValueError(f"planned.csv row {row['qid']} does not follow the frozen original-condition order")
        if row["status"] != call["status"]:
            raise ValueError(f"{row['qid']}: planned status differs from the frozen call")
        if call["status"] == "runnable":
            if (row["frozen_prompt_sha256"], row["gray_png_sha256"], int(row["slots"])) != (
                    call["prompt_sha256"], PROTOCOL_SHA256["gray.png"], SLOTS):
                raise ValueError(f"{row['qid']}: planned prompt, image or slot count differs")
            if len(call["prompt"]["frames"]) != SLOTS:
                raise ValueError(f"{row['qid']}: frozen prompt does not hold six image slots")
        elif (row["frozen_prompt_sha256"], row["gray_png_sha256"], int(row["slots"])) != ("", "", 0):
            raise ValueError(f"{row['qid']}: a technical miss must have no prompt, image or slot")
        keys.append(calls[frozen.key_of(call)])
    if sum(call["status"] == "runnable" for call in keys) != RUNNABLE_KEYS:
        raise ValueError("Expected 22 runnable keys")

    intact_bytes = (EXTENSION_RESULTS / "replies.jsonl").read_bytes()
    attempts_bytes = (EXTENSION_RESULTS / "attempts.jsonl").read_bytes()
    if sha256(intact_bytes).hexdigest() != INTACT_REPLIES_SHA256:
        raise ValueError("results_1124/replies.jsonl differs from the intact replies of 3df02a8")
    if sha256(attempts_bytes).hexdigest() != INTACT_ATTEMPTS_SHA256:
        raise ValueError("results_1124/attempts.jsonl differs from the intact ledger of 3df02a8")
    return {
        "keys": keys,
        "recordings": extension["recordings"],
        "gray": gray,
        "intact_replies": frozen.read_jsonl(intact_bytes),
        "intact_attempts": frozen.read_jsonl(attempts_bytes),
        "fingerprints": {
            "protocol": dict(PROTOCOL_SHA256),
            "extension_frozen_inputs": extension["fingerprints"],
            "intact_replies": INTACT_REPLIES_SHA256,
            "intact_attempts": INTACT_ATTEMPTS_SHA256,
        },
    }
