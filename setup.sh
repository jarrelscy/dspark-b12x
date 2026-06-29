#!/usr/bin/env bash
# Build the DSpark-on-b12x runtime: extract the image's vLLM, apply the DSpark
# overlay, and extract the real libcudart. Idempotent.
set -euo pipefail
cd "$(dirname "$0")"

IMG="voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-pr20-cu132"
VLLM_DST="./vllm-patched"

echo "==> Pulling b12x image (large, ~36GB)…"
docker pull "$IMG"

echo "==> Extracting image vLLM to ${VLLM_DST} …"
cid=$(docker create "$IMG")
rm -rf "$VLLM_DST"
docker cp "$cid:/opt/venv/lib/python3.12/site-packages/vllm" "$VLLM_DST.tmp"
mv "$VLLM_DST.tmp/vllm" "$VLLM_DST" 2>/dev/null || mv "$VLLM_DST.tmp" "$VLLM_DST"
echo "==> Extracting real libcudart.so.13 …"
docker cp "$cid:/usr/local/cuda-13.2/targets/x86_64-linux/lib/libcudart.so.13.2.75" ./libcudart.so.13
docker rm "$cid" >/dev/null

echo "==> Applying DSpark overlay (11 changed files) over the extracted vLLM …"
cp -rv overlay/vllm/* "$VLLM_DST/"

mkdir -p triton_cache
echo "==> Done. Now: VLLM_API_KEY=... docker compose -f docker-compose.dspark-b12x.yaml up -d"
echo "    Model must be cached under \$HF_HOME/hub (deepseek-ai/DeepSeek-V4-Flash-DSpark)."
