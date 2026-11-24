"""Fetch requested public Video-MME members to a local cache; validate ZIP CRC and size."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import shutil
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import zlib

def extract(member, directory):
    directory.mkdir(exist_ok=True, parents=True)
    media_id = member['media_id']
    target = directory / (media_id + ".mp4")
    if target.exists():
        crc = 0
        with target.open('rb') as video:
            for block in iter(lambda:video.read(1024*1024), b''):
                crc = zlib.crc32(block,crc)
        if target.stat().st_size == int(member['size']) and crc == int(member['crc']):
            print(media_id,'already verified'); return
        raise ValueError(f"Existing cache member {target} failed verification; choose a fresh directory.")
    n=int(member['compressed']); start=int(member['offset'])
    with tempfile.TemporaryDirectory(dir=directory) as temp:
        entry = Path(temp)/"entry"
        video = Path(temp)/"decoded.mp4"
        # Public media members can exceed 400 MB. Short independent byte ranges
        # avoid losing the whole member if one CDN connection is slow.
        span = 16 * 1024 * 1024
        last = start + n + 1023
        def download_part(left):
            right = min(left + span - 1, last)
            piece = Path(temp) / f"range-{left}"
            size = right - left + 1
            url = member['archive_url'] + f"&segment={left}"
            subprocess.run(['curl', '-fsSL', '--retry', '2', '--retry-delay', '1',
                            '--max-time', '150', '--max-filesize', str(size),
                            '--range', f'{left}-{right}', url, '-o', str(piece)], check=True)
            if piece.stat().st_size != size:
                raise EOFError('Truncated archive byte range')
            return piece
        with ThreadPoolExecutor(max_workers=6) as pool, entry.open('wb') as joined:
            for piece in pool.map(download_part, range(start, last + 1, span)):
                with piece.open('rb') as stream:
                    shutil.copyfileobj(stream, joined)
                piece.unlink()
        crc=0
        with entry.open('rb') as src, video.open('wb') as dst:
            hdr=src.read(30)
            if hdr[:4] != b'PK\x03\x04': raise ValueError('Range did not start at member header')
            a,b=struct.unpack_from('<HH',hdr,26)
            if src.read(a).decode()!=f'data/{media_id}.mp4': raise ValueError('Unexpected member name')
            src.read(b)
            method=int(member['method'])
            if method not in (0,8): raise ValueError('Unknown compression method')
            codec=zlib.decompressobj(-15) if method==8 else None
            remaining=n
            while remaining:
                block=src.read(min(1024*1024,remaining))
                if not block: raise EOFError('Truncated archive range')
                remaining -= len(block)
                decoded=codec.decompress(block) if codec else block
                dst.write(decoded); crc=zlib.crc32(decoded,crc)
            if codec:
                decoded=codec.flush();dst.write(decoded);crc=zlib.crc32(decoded,crc)
        if crc != int(member['crc']) or video.stat().st_size != int(member['size']):
            raise ValueError('ZIP checksum or size mismatch')
        os.replace(video,target)
    print(media_id,'verified',target.stat().st_size,flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache',type=Path);parser.add_argument('video_ids',nargs='+')
    args=parser.parse_args()
    with Path(__file__).with_name('media_members.csv').open() as f:
        by_id={item['video_id']:item for item in csv.DictReader(f)}
    for video in args.video_ids:extract(by_id[video],args.cache)
