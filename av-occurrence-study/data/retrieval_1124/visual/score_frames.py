"""Compute frozen CLIP query/frame cosines on the full video, without reading annotations or audio."""
import argparse
from bisect import bisect_left
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

os.environ.update({"USE_TF":"0", "USE_FLAX":"0", "HF_HUB_OFFLINE":"1", "TOKENIZERS_PARALLELISM":"false"})
import torch
import torch.nn.functional as F
import transformers
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

HERE = Path(__file__).resolve().parent


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            sha.update(block)
    return sha.hexdigest()


def nearest(pts, target):
    ix = bisect_left(pts, target)
    return min(range(max(0,ix-1), min(ix+1,len(pts))), key=lambda i: (abs(pts[i]-target), i))


def run(media, checkpoint, output):
    rules = json.loads((HERE / "rules.json").read_text())
    start = time.perf_counter()
    if digest(media) != rules["original_sha256"] or digest(checkpoint / "pytorch_model.bin") != rules["weight_sha256"]:
        raise ValueError("Source or checkpoint differs from the pinned SHA-256.")
    probe = json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0",
        "-show_packets","-show_entries","packet=pts_time:stream=width,height,duration","-of","json",str(media)]))
    stream, = probe["streams"]
    duration = float(stream["duration"])
    times = sorted(float(x["pts_time"]) for x in probe["packets"])
    width, height = stream["width"], stream["height"]
    steps = math.ceil((duration - rules["grid_seconds"])/rules["stride_seconds"])
    requested = sorted({i * rules["stride_seconds"] + offset for i in range(steps+1)
                         for offset in rules["frame_offsets_seconds"]})
    assert requested[-1] < duration
    indices = [nearest(times, t) for t in requested]
    assert len(indices) == len(set(indices))
    torch.set_num_threads(6)
    model = CLIPModel.from_pretrained(checkpoint, local_files_only=True).eval().float()
    processor = CLIPProcessor.from_pretrained(checkpoint, local_files_only=True, use_fast=False)
    with torch.inference_mode():
        query = F.normalize(model.get_text_features(**processor(text=[rules["query"]], return_tensors="pt")),dim=-1)
        loaded = time.perf_counter()
        # Decode B-frames in presentation order and retain exactly the sampled original RGB frames.
        groups = [f"eq(n\\,{i})" for i in indices]
        while len(groups) > 1:
            groups = ["(" + "+".join(groups[i:i+2]) + ")" for i in range(0,len(groups),2)]
        select = "select=" + groups[0]
        command = ["ffmpeg","-nostdin","-loglevel","error","-i",str(media),"-map","0:v:0",
                   "-vf",select,"-fps_mode","passthrough","-pix_fmt","rgb24","-f","rawvideo","pipe:1"]
        batch_size = 4
        scores = []
        with subprocess.Popen(command,stdout=subprocess.PIPE) as decoder:
            for first in range(0,len(indices),batch_size):
                images = []
                for pos in range(first,min(first+batch_size,len(indices))):
                    data = decoder.stdout.read(width*height*3)
                    if len(data) != width*height*3:
                        raise SystemExit(f"Incomplete decoded RGB frame at {requested[pos]}")
                    images.append(Image.frombytes("RGB",(width,height),data))
                pixels = processor(images=images,return_tensors="pt")["pixel_values"]
                features = F.normalize(model.get_image_features(pixel_values=pixels),dim=-1)
                scores.extend((features @ query.T).flatten().tolist())
                if len(scores) % 64 == 0:
                    print(f"{len(scores)}/{len(indices)} distinct frames",flush=True)
            if decoder.stdout.read(1) or decoder.wait() != 0:
                raise RuntimeError("Unexpected FFmpeg frame count or status.")
    complete = time.perf_counter()
    output.mkdir(parents=True,exist_ok=True)
    with (output/"frame_cosines.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["requested_s","decoded_frame_index","frame_pts_s","clip_cosine"])
        writer.writerows((target,i,times[i],score) for target,i,score in zip(requested,indices,scores))
    metrics = {"video_duration_s":duration,"grid_windows":steps+1,"unique_frames":len(scores),"threads":6,
               "device":"CPU","dtype":"float32","batch_size":batch_size,"torch":torch.__version__,
               "transformers":transformers.__version__,"input_check_model_loading_query_s":round(loaded-start,3),
               "frame_decode_preprocessing_inference_s":round(complete-loaded,3),
               "max_abs_frame_time_error_s":round(max(abs(t-times[i]) for t,i in zip(requested,indices)),6)}
    (output/"runtime.json").write_text(json.dumps(metrics,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("media",type=Path); p.add_argument("checkpoint",type=Path)
    p.add_argument("--output",type=Path,default=HERE)
    args=p.parse_args()
    run(args.media,args.checkpoint,args.output)
