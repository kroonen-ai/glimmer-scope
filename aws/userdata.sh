#!/bin/bash
# Glimmer Scope GPU box: code, layer 12 run and Glimmer weights, laid out like the Modal volumes.
exec > /var/log/glimmer-setup.log 2>&1
set -euxo pipefail
B=${GLIMMER_BUCKET}
RUN=layer12-64k-k64
mkdir -p /scope/$RUN/final /weights /opt/glimmer-scope
chown -R ubuntu:ubuntu /scope /weights /opt/glimmer-scope
sudo -u ubuntu bash <<'INNER'
set -euxo pipefail
B=${GLIMMER_BUCKET}
RUN=layer12-64k-k64
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 /opt/glimmer-scope/.venv
VIRTUAL_ENV=/opt/glimmer-scope/.venv uv pip install "torch>=2.8" "transformers==5.21.0" accelerate datasets safetensors "huggingface_hub[hf_transfer]" numpy modal
aws s3 cp --region us-east-1 s3://$B/code/glimmer-scope.tgz /tmp/code.tgz
tar -xzf /tmp/code.tgz -C /opt
aws s3 cp --region us-east-1 s3://$B/runs/$RUN/labels.jsonl /scope/$RUN/
aws s3 cp --region us-east-1 s3://$B/runs/$RUN/catalog.jsonl.gz /scope/$RUN/
aws s3 cp --region us-east-1 s3://$B/runs/$RUN/final/cfg.json /scope/$RUN/final/
HF_XET_HIGH_PERFORMANCE=1 /opt/glimmer-scope/.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("meta-models/Muse-Glimmer-30B", local_dir="/weights/Muse-Glimmer-30B",
                  allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "tokenizer*"])
PY
# the SAE may still be on its way up from the Mac
until [ "$(aws s3api head-object --region us-east-1 --bucket $B --key runs/$RUN/final/sae.safetensors --query ContentLength --output text 2>/dev/null)" = "3489950024" ]; do sleep 30; done
aws s3 cp --region us-east-1 s3://$B/runs/$RUN/final/sae.safetensors /scope/$RUN/final/
# results go to S3 every five minutes, so a spot stop loses little
echo "*/5 * * * * aws s3 sync --region us-east-1 /scope s3://$B/runs --exclude '*/checkpoint/*' --exclude 'data/*' --only-show-errors" | crontab -
INNER
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
du -sh /weights/Muse-Glimmer-30B /scope/layer12-64k-k64
# Backstop: with the GPU idle for 30 minutes, copy results to S3 and power off; the instance is
# launched to terminate on shutdown, so nothing keeps billing.
cat > /usr/local/bin/glimmer-idle-guard.sh <<'GUARD'
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
GUARD
chmod +x /usr/local/bin/glimmer-idle-guard.sh
echo "*/5 * * * * root /usr/local/bin/glimmer-idle-guard.sh" > /etc/cron.d/glimmer-idle-guard
touch /opt/glimmer-scope/READY
