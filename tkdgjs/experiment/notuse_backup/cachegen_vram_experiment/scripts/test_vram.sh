#!/bin/bash
#===============================================================================
# Test VRAM with a single request
#
# Usage: ./test_vram.sh [prompt_tokens] [max_tokens]
#        Default: prompt_tokens=256, max_tokens=32
#===============================================================================

PROMPT_TOKENS=${1:-256}
MAX_TOKENS=${2:-32}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"

echo "========================================"
echo "VRAM Test"
echo "========================================"
echo "Prompt tokens: $PROMPT_TOKENS"
echo "Max tokens: $MAX_TOKENS"
echo ""

# Check if vLLM is running
if ! curl -s http://localhost:8000/v1/models &> /dev/null; then
    echo "ERROR: vLLM is not running"
    echo "Start vLLM first with ./start_vllm_native.sh or ./start_vllm_cachegen.sh"
    exit 1
fi

# Get current mode
KV_BACKEND=$(curl -s http://localhost:8000/metrics | grep -o 'kv_offloading_backend="[^"]*"' | cut -d'"' -f2)
echo "Mode: KV Offloading = $KV_BACKEND"

# Clear disk cache
rm -rf /tmp/lmcache/lmcache_disk/*

# VRAM before
echo ""
echo "=== VRAM Before Request ==="
VRAM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "VRAM: $VRAM_BEFORE MB"

# Generate prompt
python3 -c "print('word ' * $((PROMPT_TOKENS / 4)))" > /tmp/prompt.txt

# Send request
echo ""
echo "Sending request..."
START_TIME=$(date +%s.%N)

RESPONSE=$(curl -s -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"meta-llama/Llama-3.2-1B-Instruct\", \"prompt\": $(cat /tmp/prompt.txt | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))'), \"max_tokens\": $MAX_TOKENS, \"temperature\": 0}")

END_TIME=$(date +%s.%N)
DURATION=$(echo "$END_TIME - $START_TIME" | bc)

# VRAM after
sleep 2
echo ""
echo "=== VRAM After Request (2s) ==="
VRAM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "VRAM: $VRAM_AFTER MB"
echo "Delta: $((VRAM_AFTER - VRAM_BEFORE)) MB"

# Wait for offload to complete
sleep 5
echo ""
echo "=== VRAM After Offload (7s) ==="
VRAM_FINAL=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "VRAM: $VRAM_FINAL MB"
echo "Delta from before: $((VRAM_FINAL - VRAM_BEFORE)) MB"

# Disk usage
echo ""
echo "=== Disk Usage ==="
DISK_SIZE=$(du -sh /tmp/lmcache/lmcache_disk/ 2>/dev/null | cut -f1)
echo "Disk: $DISK_SIZE"

# Check LMCache logs
echo ""
echo "=== LMCache Store Log ==="
LOG_FILE="$LOG_DIR/vllm_native.log"
if [ -f "$LOG_FILE" ]; then
    grep "Stored" "$LOG_FILE" | tail -1
else
    echo "Log file not found"
fi

echo ""
echo "=== Summary ==="
echo "Duration: ${DURATION}s"
echo "VRAM Before: $VRAM_BEFORE MB"
echo "VRAM After: $VRAM_AFTER MB"
echo "VRAM Delta: $((VRAM_AFTER - VRAM_BEFORE)) MB"
echo "Disk Size: $DISK_SIZE"
