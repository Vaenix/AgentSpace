#!/usr/bin/env python3
"""Score paired video questions using frozen, verified evidence intervals."""

import argparse
from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path


IOU_THRESHOLD = 0.5


@dataclass(frozen=True)
class Reference:
    pair_id: str
    video_id: str
    qid: str
    occurrence: int
    answer_id: str
    interval_s: tuple[float, float]


@dataclass(frozen=True)
class Prediction:
    answer_id: str | None
    interval_s: tuple[float, float] | None


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            label = f"{path}:{line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label} 含有无效 JSON。{exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{label} 每行须为 JSON 对象。")
            yield row, label


def nonempty_string(row, key, label):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} 的 {key} 需要无首尾空白的非空字符串。")
    return value


def interval(value, label):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} 的 interval_s 需要包含两个端点。")
    if any(type(point) not in (int, float) or not math.isfinite(point) for point in value):
        raise ValueError(f"{label} 的 interval_s 需要有限数值端点。")
    start, end = value
    if not 0 <= start < end:
        raise ValueError(f"{label} 的 interval_s 需要满足 0 ≤ start < end。")
    return start, end


def load_gold(path):
    by_pair = {}
    for row, label in jsonl(path):
        if row.get("review_status") != "verified":
            raise ValueError(f"{label} 需要 review_status=verified 才能评分。")
        occurrence = row.get("occurrence")
        if type(occurrence) is not int or occurrence not in (1, 2):
            raise ValueError(f"{label} 的 occurrence 需要是整数 1 或 2。")
        reference = Reference(
            pair_id=nonempty_string(row, "pair_id", label),
            video_id=nonempty_string(row, "video_id", label),
            qid=nonempty_string(row, "qid", label),
            occurrence=occurrence,
            answer_id=nonempty_string(row, "answer_id", label),
            interval_s=interval(row.get("interval_s"), label),
        )
        pair = by_pair.setdefault(reference.pair_id, {})
        if occurrence in pair:
            raise ValueError(f"{label} 的题对 {reference.pair_id} 重复标注第 {occurrence} 次事件。")
        pair[occurrence] = reference

    references = []
    for pair_id, members in sorted(by_pair.items()):
        if set(members) != {1, 2}:
            raise ValueError(f"题对 {pair_id} 必须包含第 1 次和第 2 次事件各一题。")
        first, second = members[1], members[2]
        if first.video_id != second.video_id or first.qid == second.qid or first.answer_id == second.answer_id:
            raise ValueError(f"题对 {pair_id} 需要同一视频、两个不同题号和两个不同答案。")
        if first.interval_s[1] > second.interval_s[0]:
            raise ValueError(f"题对 {pair_id} 的第 1 次和第 2 次事件需要按顺序排列且互不重叠。")
        references.append((first, second))
    return references


def load_predictions(path, references, conditions):
    predictions = {name: {} for name in conditions}
    if path is None:
        return predictions
    valid_keys = {(item.pair_id, item.qid) for pair in references for item in pair}

    for row, label in jsonl(path):
        condition = nonempty_string(row, "condition", label)
        if conditions and condition not in predictions:
            raise ValueError(f"{label} 的条件 {condition} 未通过 --condition 声明。")
        key = (nonempty_string(row, "pair_id", label), nonempty_string(row, "qid", label))
        if references and key not in valid_keys:
            raise ValueError(f"{label} 的题号 {key!r} 不在冻结题单中。")
        if "answer_id" not in row:
            raise ValueError(f"{label} 缺少 answer_id。拒答请显式填入 null。")
        answer_id = None if row["answer_id"] is None else nonempty_string(row, "answer_id", label)
        predicted_interval = None if row.get("interval_s") is None else interval(row["interval_s"], label)
        group = predictions.setdefault(condition, {})
        if key in group:
            raise ValueError(f"{label} 在条件 {condition} 下重复预测题号 {key!r}。")
        group[key] = Prediction(answer_id, predicted_interval)
    return predictions


def iou(left, right):
    overlap = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    if overlap == 0:
        return 0.0
    return overlap / (max(left[1], right[1]) - min(left[0], right[0]))


def score_question(target, other, prediction):
    answer = prediction.answer_id if prediction is not None else None
    predicted_interval = prediction.interval_s if prediction is not None else None
    target_iou = other_iou = None
    if predicted_interval is not None:
        target_iou = iou(predicted_interval, target.interval_s)
        other_iou = iou(predicted_interval, other.interval_s)
    correct_interval = target_iou is not None and target_iou >= IOU_THRESHOLD
    # An interval spanning both anchors equally does not identify either occurrence.
    ambiguous = correct_interval and target_iou == other_iou
    return {
        "qid": target.qid,
        "occurrence": target.occurrence,
        "prediction_present": prediction is not None,
        "abstained": prediction is not None and answer is None,
        "answer_id": answer,
        "interval_s": predicted_interval,
        "answer_correct": answer == target.answer_id,
        "answer_is_other_occurrence": answer == other.answer_id,
        "target_iou": target_iou,
        "other_iou": other_iou,
        "localization_correct": correct_interval and not ambiguous,
        "localized_other_occurrence": other_iou is not None and other_iou >= IOU_THRESHOLD and other_iou > target_iou,
        "ambiguous_localization": ambiguous,
    }


def score_pair(pair, predictions):
    answers = [
        score_question(target, other, predictions.get((target.pair_id, target.qid)))
        for target, other in (pair, reversed(pair))
    ]
    answer_swap = all(item["answer_is_other_occurrence"] for item in answers)
    localization_swap = all(item["localized_other_occurrence"] for item in answers)
    return {
        "pair_id": pair[0].pair_id,
        "video_id": pair[0].video_id,
        "joint_pair_success": all(item["answer_correct"] and item["localization_correct"] for item in answers),
        "answer_swap": answer_swap,
        "localization_swap": localization_swap,
        "full_swap": answer_swap and localization_swap,
        "questions": answers,
    }


def metric(rows, flag):
    count = sum(row[flag] for row in rows)
    return {"count": count, "total": len(rows), "rate": count / len(rows)}


def score_group(references, predictions):
    expected = 2 * len(references)
    report = {"expected_questions": expected, "submitted_questions": len(predictions)}
    if not references or not predictions:
        return dict(report, status="未跑", metrics=None, pairs=[])
    pairs = [score_pair(pair, predictions) for pair in references]
    questions = [item for pair in pairs for item in pair["questions"]]
    question_flags = [
        "prediction_present", "abstained", "answer_correct", "localization_correct",
        "answer_is_other_occurrence", "localized_other_occurrence", "ambiguous_localization",
    ]
    pair_flags = ["joint_pair_success", "answer_swap", "localization_swap", "full_swap"]
    metrics = {flag: metric(questions, flag) for flag in question_flags}
    metrics.update({flag: metric(pairs, flag) for flag in pair_flags})
    return dict(
        report, status="已评分" if len(predictions) == expected else "部分完成", metrics=metrics, pairs=pairs,
    )


def compare_groups(groups):
    comparisons = []
    for (left_name, left), (right_name, right) in combinations(groups.items(), 2):
        result = {"left_condition": left_name, "right_condition": right_name}
        if left["status"] != "已评分" or right["status"] != "已评分":
            comparisons.append(dict(result, status="待完成", joint_pair_comparison=None))
            continue
        scores = zip(left["pairs"], right["pairs"], strict=True)
        changes = [int(b["joint_pair_success"]) - int(a["joint_pair_success"]) for a, b in scores]
        comparisons.append(dict(result, status="已比较", joint_pair_comparison={
            "left_only_success": changes.count(-1),
            "right_only_success": changes.count(1),
            "same_pair_outcome": changes.count(0),
            "total_pairs": len(changes),
            "right_minus_left_percentage_points": 100 * sum(changes) / len(changes),
        }))
    return comparisons


def main():
    parser = argparse.ArgumentParser(description="按同一冻结题单评分重复声画事件题对。")
    parser.add_argument("--gold", type=Path, required=True, help="已复核标准答案 JSONL")
    parser.add_argument("--predictions", type=Path, help="主体实验预测 JSONL")
    parser.add_argument("--condition", action="append", default=[], help="声明实验条件，可重复使用")
    args = parser.parse_args()

    try:
        if any(not value.strip() or value != value.strip() for value in args.condition):
            raise ValueError("--condition 需要无首尾空白的非空字符串。")
        if len(set(args.condition)) != len(args.condition):
            raise ValueError("--condition 不能重复声明同一条件。")
        references = load_gold(args.gold)
        predictions = load_predictions(args.predictions, references, args.condition)
        groups = {name: score_group(references, rows) for name, rows in predictions.items()}
        if not any(group["metrics"] is not None for group in groups.values()):
            status = "未跑"
        else:
            status = "已评分" if all(group["status"] == "已评分" for group in groups.values()) else "部分完成"
        print(json.dumps({
            "status": status,
            "gold_pairs": len(references),
            "gold_questions": 2 * len(references),
            "iou_threshold": IOU_THRESHOLD,
            "conditions": groups,
            "comparisons": compare_groups(groups),
        }, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, OverflowError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
