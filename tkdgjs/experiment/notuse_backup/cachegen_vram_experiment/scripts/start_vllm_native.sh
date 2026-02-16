#!/bin/bash
#===============================================================================
# Start vLLM with Native (torch) serialization - No compression
#
# Usage: ./start_vllm_native.sh
#===============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"

# Environment
export PATH=/home/noslab-gpu/tkdgjs/tkdgjs/bin:$PATH
export LMCACHE_CONFIG_FILE="$PROJECT_DIR/configs/lmcache_native.yaml"
export PYTHONHASHSEED=0
export VLLM_ATTENTION_BACKEND=TRITON_ATTN

# vLLM parameters
MODEL="meta-llama/Llama-3.2-1B-Instruct"
PORT=8000
KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"discard_partial_chunks":false}}'

echo "========================================"
echo "Starting vLLM with Native mode"
echo "========================================"
echo "Model: $MODEL"
echo "Config: $LMCACHE_CONFIG_FILE"
echo "Log: $LOG_DIR/vllm_native.log"
echo ""

# Clear disk cache
echo "Clearing disk cache..."
rm -rf /tmp/lmcache/lmcache_disk/*

# Start vLLM
nohup vllm serve "$MODEL" \
  --port "$PORT" \
  --dtype half \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 131072 \
  --gpu-memory-utilization 0.7 \
  --disable-hybrid-kv-cache-manager \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  --kv-offloading-backend lmcache \
  --kv-offloading-size 4 \
  --scheduling-policy fcfs \
  --enable-chunked-prefill \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}' \
  --enable-mfu-metrics \
  --enable-logging-iteration-details \
  --attention-config '{"backend": "TRITON_ATTN"}' \
  > "$LOG_DIR/vllm_native.log" 2>&1 &

VLLM_PID=$!
echo "vLLM started with PID: $VLLM_PID"

# Wait for vLLM to be ready
echo "Waiting for vLLM to be ready..."
for i in {1..40}; do
    if curl -s http://localhost:$PORT/v1/models &> /dev/null; then
        echo "vLLM is ready!"
        
        # Verify LMCache connection
        sleep 2
        KV_BACKEND=$(curl -s http://localhost:$PORT/metrics | grep -o 'kv_offloading_backend="[^"]*"' | cut -d'"' -f2)
        echo "KV Offloading Backend: $KV_BACKEND"
        
        if [ "$KV_BACKEND" = "lmcache" ]; then
            echo "SUCCESS: LMCache connected!"
        else
            echo "WARNING: LMCache may not be connected"
        fi
        exit 0
    fi
    echo "  Waiting... ($i/40)"
    sleep 3
done

echo "ERROR: vLLM failed to start"
echo "Check log: $LOG_DIR/vllm_native.log"
tail -30 "$LOG_DIR/vllm_native.log"
exit 1
