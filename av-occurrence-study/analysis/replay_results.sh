#!/usr/bin/env bash
# Usage: bash replay_results.sh /path/to/AgentSpace
# The clone must contain the linked study commits and their history.
set -euo pipefail
repo=${1:?"Supply the path to the AgentSpace clone"}
export GIT_DIR
GIT_DIR=$(git -C "$repo" rev-parse --absolute-git-dir)
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT

replay() {
    local label=$1 commit=$2 folder=$3 script=$4 expected=$5
    local stage="$work/$label"
    mkdir "$stage"
    git archive "$commit" av-occurrence-study/run | tar -x -C "$stage"
    local batch="$stage/av-occurrence-study/run/$folder"
    python3 "$batch/$script" --output "$stage/replayed.json"
    cmp "$batch/$expected" "$stage/replayed.json"
    printf '%s: byte-identical to %s at %.8s\n' "$label" "$expected" "$commit"
}

replay development 87ee601846c9062ec7fc98a206b633f83b0b7d28 replication_1124 replay_parse.py parse_replay_matched_seconds.json
replay held-out 89fc8a5a8a4dd7ee0574663e4566e96224ce524c confirmation_1124 summarize.py statistics.json
replay extension 3df02a8e1b84d876ebdf8b54b9826d803860fcf1 participant_extension/results_1124 summarize.py statistics.json
replay gray c79f314327b129e4f05ba31f0c04438a46676711 extension_gray_control summarize_gray.py statistics.json
replay crossed 67f2d70a6a36bfafdbca36027a0731d269b398c1 extension_crossed_labels summarize_crossed.py statistics.json
replay frame-swap cb2b9ab601d6dacc148ca60640939f895fb06acf extension_frame_swap summarize_swap.py statistics.json
