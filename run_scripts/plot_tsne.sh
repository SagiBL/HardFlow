#!/bin/bash
# Script to run t-SNE visualization on planned waypoints across all eval folders.

eval_dir="/home/linux/test_project/HardFlow/logs/maze2d-large-v1/eval"

# If a custom path is supplied as first argument, use it
if [ ! -z "$1" ]; then
    eval_dir="$1"
fi

echo "=== Running t-SNE on planned waypoints in: $eval_dir ==="

python run/plot_tsne.py --eval_dir "$eval_dir"
