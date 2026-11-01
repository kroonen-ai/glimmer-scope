#!/bin/bash
# run.sh NAME ARGS...: one run_here.py GPU step, logged under /scope/logs
mkdir -p /scope/logs
cd /opt/glimmer-scope
# GPU work only on this box: no Meta key here, steered replies are kept unjudged
export NO_JUDGE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
name=$1; shift
.venv/bin/python scripts/run_here.py "$@" 2>&1 | grep --line-buffered -v "max_new_tokens\|Loading weights" > /scope/logs/$name.log
