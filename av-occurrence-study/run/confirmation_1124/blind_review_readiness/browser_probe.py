"""Try the batch originals in Playwright browser engines, as a reviewer's player would.

For each pair in the frozen queue, a muted <video> element loads the official URL,
plays from 0 s for --play seconds of wall time, then seeks to a few seconds before
the second candidate sound and plays again. The report records metadata, codec
support, the playback time gained per wall second, stalls ("waiting" events) and
media errors. Playwright engines approximate but are not Safari or Google Chrome.

    python3 av-occurrence-study/run/confirmation_1124/blind_review_readiness/browser_probe.py \
        --output /tmp/blind-review-browsers.json [--engine webkit --engine chromium]
"""

import argparse
import asyncio
import json
from pathlib import Path
import time

from playwright.async_api import async_playwright

QUEUE = Path(__file__).resolve().parent.parent / "manifest" / "selected.jsonl"

PROBE = r"""
async ([url, segments, playSeconds]) => {
  const types = {hevc: 'video/mp4; codecs="hvc1.1.6.L120.90"', avc: 'video/mp4; codecs="avc1.640028"',
                 aac: 'audio/mp4; codecs="mp4a.40.2"'};
  const can = Object.fromEntries(Object.entries(types).map(([k, v]) => [k, document.createElement("video").canPlayType(v)]));
  const v = document.createElement("video");
  v.muted = true; v.preload = "auto"; document.body.append(v);
  let waiting = 0; v.addEventListener("waiting", () => waiting++);
  const once = (event, ms) => new Promise(done => {
    const timer = setTimeout(() => done("timeout"), ms);
    v.addEventListener(event, () => { clearTimeout(timer); done("ok"); }, {once: true});
    v.addEventListener("error", () => { clearTimeout(timer); done("error"); }, {once: true});
  });
  const error = () => v.error ? {code: v.error.code, message: v.error.message} : null;
  const out = {can};
  const t0 = performance.now();
  v.src = url;
  out.metadata = await once("loadedmetadata", 180000);
  out.metadata_wall_s = (performance.now() - t0) / 1000;
  out.duration_s = Number.isFinite(v.duration) ? v.duration : null; out.width = v.videoWidth; out.height = v.videoHeight;
  out.audio_tracks = v.audioTracks ? v.audioTracks.length : null;
  out.error = error();
  out.segments = [];
  if (out.metadata !== "ok") return out;
  for (const start of segments) {
    const seg = {start_s: start};
    if (Math.abs(v.currentTime - start) > 0.01) {
      const s0 = performance.now();
      v.currentTime = start;
      seg.seek = await once("seeked", 180000);
      seg.seek_wall_s = (performance.now() - s0) / 1000;
    }
    waiting = 0;
    const from = v.currentTime;
    try { await v.play(); seg.play = "ok"; } catch (e) { seg.play = String(e); }
    await new Promise(done => setTimeout(done, playSeconds * 1000));
    v.pause();
    seg.media_advanced_s = v.currentTime - from;
    seg.wall_s = playSeconds;
    seg.realtime_ratio = seg.media_advanced_s / playSeconds;
    seg.waiting_events = waiting;
    seg.buffered = Array.from({length: v.buffered.length}, (_, i) => [v.buffered.start(i), v.buffered.end(i)]);
    seg.decoded_frames = v.getVideoPlaybackQuality ? v.getVideoPlaybackQuality().totalVideoFrames : null;
    seg.error = error();
    out.segments.push(seg);
  }
  return out;
}
"""


async def main(args):
    pairs = [json.loads(line) for line in QUEUE.read_text().splitlines() if line.strip()]
    report = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "play_seconds": args.play,
              "engines": {}}
    async with async_playwright() as playwright:
        for name in args.engine:
            entry = report["engines"][name] = {}
            try:
                browser = await getattr(playwright, name).launch(timeout=120000)
            except Exception as error:
                entry["launch_error"] = str(error).splitlines()[0]
                continue
            entry["version"] = browser.version
            for pair in pairs:
                second = min(o["sound_s"][0] for o in pair["occurrences"] if o["ordinal"] == 2)
                page = await browser.new_page()
                try:
                    entry[pair["pair_id"]] = await page.evaluate(
                        PROBE, [pair["media_url"], [0.0, max(0.0, round(second - 5, 3))], args.play])
                except Exception as error:
                    entry[pair["pair_id"]] = {"exception": str(error).splitlines()[0]}
                await page.close()
                print(name, pair["pair_id"], json.dumps(entry[pair["pair_id"]]), flush=True)
            await browser.close()
    args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", action="append", choices=["webkit", "chromium", "firefox"])
    parser.add_argument("--play", type=float, default=30.0, help="wall seconds of playback per segment")
    parsed = parser.parse_args()
    parsed.engine = parsed.engine or ["webkit", "chromium", "firefox"]
    asyncio.run(main(parsed))
