"""Summarize optional macOS SoundAnalysis outputs in the media cache as machine leads."""
import argparse
import csv
from pathlib import Path

FAMILIES = {
    '观众掌声': {'applause', 'clapping'},
    '球场哨声': {'whistle', 'whistling', 'sports_whistle', 'referee_whistle'},
    '警长敲钟': {'bell', 'chime', 'bicycle_bell', 'cowbell'},
    '微波炉完成提示音': {'beep', 'microwave_oven', 'bell', 'bicycle_bell', 'chime'},
    '泼水声': {'liquid_splashing', 'splash', 'splashing'},
    '入水／溅水声': {'liquid_splashing', 'splash', 'splashing'},
}

def intervals(path, labels):
    hits = []
    with path.open() as stream:
        for begin, duration, label, confidence in csv.reader(stream, delimiter='\t'):
            start = float(begin)
            end = start + float(duration)
            score = float(confidence)
            if label not in labels or score < 0.5:
                continue
            if hits and start <= hits[-1][1]:
                head = hits[-1]
                head[1] = max(head[1], end)
                if score > head[3]:
                    head[2:4] = [label, score]
                head[4].add(label)
            else:
                hits.append([start, end, label, score, {label}])
    return hits

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with Path(__file__).with_name('candidates.csv').open() as src, args.output.open('w') as dst:
        writer = csv.writer(dst)
        writer.writerow(['video_id','cue','status','start_sec','end_sec','peak_label','peak_confidence','matched_labels'])
        for row in csv.DictReader(src):
            video = row['video_id']
            cue = row['cue']
            path = args.cache / (video.removeprefix('video') + '-sound.tsv')
            if not path.exists():
                writer.writerow([video, cue, '未跑', '', '', '', '', ''])
                continue
            hits = intervals(path, FAMILIES[cue])
            if not hits:
                writer.writerow([video, cue, '预定标签无过阈值窗口', '', '', '', '', ''])
            for start, end, label, score, names in hits:
                writer.writerow([video, cue, '机器候选声音', start, end, label, score, '|'.join(sorted(names))])
