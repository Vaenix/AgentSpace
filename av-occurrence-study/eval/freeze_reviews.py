#!/usr/bin/env python3
"""Prepare a human adjudication table and freeze its explicitly included pairs."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import unicodedata

from score import load_gold


IDENTITY = ["题对", "视频", "材料指纹"]
EVIDENCE = ["复核人 A", "复核人 B"]
DECISION = ["决策", "仲裁理由", "声音类别", "发声1秒数", "动作1秒数", "动作1", "发声2秒数", "动作2秒数", "动作2", "干扰动作1", "干扰动作2"]
DISPOSITIONS = {"propose-include": "建议纳入", "uncertain": "不确定", "exclude": "排除"}
JUDGMENTS = {
    "independent": "同类且独立",
    "first_second": "确为前两次",
    "actions_differ": "动作明确且不同",
    "answerable_without_audio": "静音仍可判定",
    "answerable_without_vision": "仅听仍可判定",
}
REVIEW_VALUES = {"yes": "是", "no": "否", "uncertain": "不确定", "unavailable": "素材不可用"}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require_text(raw, label):
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label}需要填写。")
    return raw.strip()


def load_materials(args):
    queue_bytes = args.queue.read_bytes()
    queue_sha = digest(queue_bytes)
    queue = [json.loads(line) for line in queue_bytes.splitlines() if line.strip()]
    videos = {row["pair_id"]: row["video_id"] for row in queue}
    if not queue or len(videos) != len(queue):
        raise ValueError("候选队列为空或题对编号重复。")
    reviewers = {}
    sources = {"queue_sha256": queue_sha}
    for expected_slot, path in [("A", args.review_a), ("B", args.review_b)]:
        raw = path.read_bytes()
        review = json.loads(raw)
        if review["record_type"] != "independent_review" or review["schema_version"] != 1 or review["queue_revision"] != queue_sha:
            raise ValueError(f"{path} 不是本批队列导出的复核记录。")
        if review["reviewer"]["slot"] != expected_slot:
            raise ValueError(f"{path} 的复核身份应为 {expected_slot}。")
        code = require_text(review["reviewer"]["code"], "复核人代号")
        records = {}
        for entry in review["records"]:
            pair_id = entry["pair_id"]
            if pair_id in records or videos.get(pair_id) != entry["video_id"]:
                raise ValueError(f"{path} 的题对 {pair_id} 重复或与候选视频不符。")
            records[pair_id] = entry
        if records.keys() != videos.keys():
            raise ValueError(f"{path} 的题单与候选队列不一致。")
        reviewers[expected_slot] = {"code": code, "records": records}
        sources[f"review_{expected_slot.lower()}_sha256"] = digest(raw)
    if reviewers["A"]["code"].casefold() == reviewers["B"]["code"].casefold():
        raise ValueError("两份独立记录需要不同的复核人代号。")
    fingerprint = digest("\n".join(sources.values()).encode())
    return queue, reviewers, sources, fingerprint


def time_note(clip, event):
    start, end = [clip.get(f"{event}_{part}_s") for part in ("start", "end")]
    return "待填写" if start is None or end is None else f"{start}~{end}"


def summarize(record):
    judgments = record["judgments"]
    lines = [DISPOSITIONS.get(judgments.get("disposition"), "待填写")]
    lines.extend(f"{label}：{REVIEW_VALUES.get(judgments.get(key), '待填写')}" for key, label in JUDGMENTS.items())
    for ordinal in ("1", "2"):
        clip = record["clips"].get(ordinal, {})
        lines.append(f"发声{ordinal}：{clip.get('sound') or '待填写'}（{time_note(clip, 'sound')}）")
        lines.append(f"动作{ordinal}：{clip.get('action') or '待填写'}（{time_note(clip, 'action')}）")
    if judgments.get("notes"):
        lines.append("依据：" + judgments["notes"])
    return "\n".join(lines)


def write_table(path, queue, reviewers, fingerprint):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        table = csv.DictWriter(stream, IDENTITY + EVIDENCE + DECISION, dialect="excel-tab")
        table.writeheader()
        for pair in queue:
            pair_id = pair["pair_id"]
            table.writerow({
                "题对": pair_id,
                "视频": pair["video_id"],
                "材料指纹": fingerprint,
                **{f"复核人 {slot}": summarize(review["records"][pair_id]) for slot, review in reviewers.items()},
            })


def parse_interval(raw, label, duration):
    try:
        start, end = map(float, raw.split("~"))
    except ValueError as exc:
        raise ValueError(f"{label}请按 12.3~13.4 的格式填写秒数。") from exc
    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= duration:
        raise ValueError(f"{label}的起止应符合 0 ≤ 开始 < 结束 ≤ 视频时长。")
    return [start, end]


def action_id(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = " ".join(normalized.split())
    return "act_" + digest(normalized.encode())[:16]


def confirmed_pair(row, source, reviews, offset):
    pair_id = source["pair_id"]
    sound_label = require_text(row["声音类别"], f"题对 {pair_id} 的声音类别")
    for review in reviews.values():
        original = review["records"][pair_id]
        judgment = original["judgments"]
        if judgment.get("disposition") not in DISPOSITIONS or not (original["clips"] or judgment.get("notes", "").strip()):
            raise ValueError(f"题对 {pair_id} 需要两位复核人留下判断及可追溯依据。")

    duration = source["duration_s"]
    if not math.isfinite(duration) or duration < 600:
        raise ValueError(f"题对 {pair_id} 的原视频至少需要 600 秒。")
    sounds, actions, answers = [], [], []
    for ordinal in (1, 2):
        sounds.append(parse_interval(row[f"发声{ordinal}秒数"], f"题对 {pair_id} 发声 {ordinal}", duration))
        actions.append(parse_interval(row[f"动作{ordinal}秒数"], f"题对 {pair_id} 动作 {ordinal}", duration))
        answers.append(require_text(row[f"动作{ordinal}"], f"题对 {pair_id} 的动作 {ordinal}"))
    if sounds[0][1] > sounds[1][0]:
        raise ValueError(f"题对 {pair_id} 的两次发声必须按时间先后排列且互不重叠。")
    if any(action[0] < sound[1] for sound, action in zip(sounds, actions)):
        raise ValueError(f"题对 {pair_id} 的动作须在对应发声结束后开始。")

    distractors = [require_text(row[f"干扰动作{index}"], f"题对 {pair_id} 的干扰动作 {index}") for index in (1, 2)]
    options = [answers[0], distractors[0], answers[1], distractors[1]]
    rotated = options[offset:] + options[:offset]
    option_map = {action_id(text): text for text in rotated}
    if len(option_map) != 4:
        raise ValueError(f"题对 {pair_id} 的四个动作选项需互不相同。")
    gold, questions = [], []
    for index, order in enumerate(("一", "二")):
        identity = {"pair_id": pair_id, "qid": f"{pair_id}-{index + 1}"}
        gold.append(dict(
            identity, video_id=source["video_id"], occurrence=index + 1, answer_id=action_id(answers[index]),
            answer_text=answers[index], interval_s=sounds[index], action_interval_s=actions[index], source_url=source["media_url"],
            review_status="verified",
        ))
        questions.append(dict(identity, question=f"在这段视频里，第{order}次{sound_label}结束后，拍摄者接下来开始做的第一个可见动作是什么？", options=option_map))
    return gold, questions


def write_jsonl(path, rows):
    raw = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows).encode()
    path.write_bytes(raw)
    return digest(raw)


def freeze(args, queue, reviews, sources, fingerprint):
    with args.adjudication.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, dialect="excel-tab")
        if reader.fieldnames != IDENTITY + EVIDENCE + DECISION:
            raise ValueError("请保留原始仲裁表的列顺序和标题，并以制表符分隔保存。")
        table = list(reader)
    if len(table) != len(queue) or len({row["题对"] for row in table}) != len(queue):
        raise ValueError("仲裁表需要保留全部原始候选，每对一行。")
    by_id = {row["题对"]: row for row in table}
    adjudicator = require_text(args.adjudicator, "仲裁人代号")
    gold, questions, decisions = [], [], []
    included_videos = set()
    for source in queue:
        pair_id = source["pair_id"]
        row = by_id.get(pair_id)
        if row is None or None in row or None in row.values() or row["视频"] != source["video_id"] or row["材料指纹"] != fingerprint:
            raise ValueError(f"题对 {pair_id} 的仲裁行与原始材料不一致。")
        submitted = row["决策"].strip()
        if submitted not in {"纳入", "排除", "待定"}:
            raise ValueError(f"题对 {pair_id} 的人工决策须填写纳入、排除或待定。")
        reason = require_text(row["仲裁理由"], f"题对 {pair_id} 的仲裁理由")
        outcomes = {slot: review["records"][pair_id]["judgments"].get("disposition") for slot, review in reviews.items()}
        decisions.append({"pair_id": pair_id, "decision": submitted, "reason": reason, "reviewer_dispositions": outcomes})
        if submitted == "纳入":
            if source["video_id"] in included_videos:
                raise ValueError(f"视频 {source['video_id']} 的纳入题对超过一对。")
            pair_gold, pair_questions = confirmed_pair(row, source, reviews, (len(gold) // 2) % 4)
            gold.extend(pair_gold)
            questions.extend(pair_questions)
            included_videos.add(source["video_id"])
    if not gold:
        raise ValueError("当前没有经人工确认纳入的题对。实验状态为未跑。")

    args.output.mkdir(exist_ok=False)
    gold_path = args.output / "gold.jsonl"
    gold_sha = write_jsonl(gold_path, gold)
    load_gold(gold_path)
    questions_sha = write_jsonl(args.output / "questions.jsonl", questions)
    manifest = {
        "adjudicator": adjudicator, "frozen_at": datetime.now(timezone.utc).isoformat(), "pairs_included": len(gold) // 2,
        "reviewer_codes": {slot: review["code"] for slot, review in reviews.items()},
        **sources, "adjudication_sha256": digest(args.adjudication.read_bytes()),
        "gold_sha256": gold_sha, "questions_sha256": questions_sha, "decisions": decisions,
    }
    (args.output / "freeze.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="对照两份独立复核并冻结人工仲裁通过的事件题对。")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "freeze"):
        command = commands.add_parser(name)
        command.add_argument("--queue", type=Path, required=True)
        command.add_argument("--review-a", type=Path, required=True)
        command.add_argument("--review-b", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    freeze_command = commands.choices["freeze"]
    freeze_command.add_argument("--adjudication", type=Path, required=True)
    freeze_command.add_argument("--adjudicator", required=True)
    args = parser.parse_args()
    try:
        queue, reviews, sources, fingerprint = load_materials(args)
        if args.command == "prepare":
            write_table(args.output, queue, reviews, fingerprint)
        else:
            freeze(args, queue, reviews, sources, fingerprint)
        print(args.output)
    except (ValueError, OSError, TypeError, KeyError, AttributeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
