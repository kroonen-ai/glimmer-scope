#!/bin/bash
# Never pay for an idle GPU: below 5% use for 30 minutes straight, back up to S3 and power off
# (the instance terminates on shutdown).
f=/var/run/glimmer-gpu-idle-since
u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
if [ "${u:-0}" -ge 5 ]; then rm -f $f; exit 0; fi
[ -f $f ] || date +%s > $f
if [ $(( $(date +%s) - $(cat $f) )) -ge 1800 ]; then
  sudo -u ubuntu aws s3 sync --region us-east-1 /scope s3://${GLIMMER_BUCKET}/runs --exclude "*/checkpoint/*" --exclude "data/*" --only-show-errors
  shutdown -h now
fi
