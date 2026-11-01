#!/bin/bash
# run.sh NAME ARGS...: one run_here.py step with the Meta key read into memory, logged under /scope/logs
mkdir -p /scope/logs
cd /opt/glimmer-scope
export META_API_KEY=$(aws ssm get-parameter --region us-east-1 --name /glimmer/meta-api-key --with-decryption --query Parameter.Value --output text)
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
name=$1; shift
.venv/bin/python scripts/run_here.py "$@" 2>&1 | grep --line-buffered -v "max_new_tokens\|Loading weights" > /scope/logs/$name.log
