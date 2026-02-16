#!/bin/bash
#===============================================================================
# Full Experiment Runner - Native vs CacheGen comparison
#
# This script runs the complete experiment comparing Native and CacheGen modes.
# It automatically starts vLLM with each mode, runs tests, and collects results.
#
# Usage: ./run_experiment.sh
#===============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
RESULTS_DIR="$PROJECT_DIR/results"

# Experiment parameters
PREFILL_SIZES=(256 512 1024 2048)
MAX_TOKENS=32
COOLDOWN=30  # seconds between experiments

echo "========================================"
echo "CacheGen VRAM Experiment Runner"
echo "========================================"
echo "Prefill sizes: ${PREFILL_SIZES[*]}"
echo "Max tokens: $MAX_TOKENS"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to run test and collect results
run_test() {
    local MODE=$1
    local PREFILL=$2
    
    echo "----------------------------------------"
    echo "Testing: MODE=$MODE, PREFILL=$PREFILL"
    echo "----------------------------------------"
    
    # Clear disk cache
    rm -rf /tmp/lmcache/lmcache_disk/*
    
    # VRAM before
    VRAM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    
    # Generate prompt
    python3 -c "print('word ' * $((PREFILL / 4)))" > /tmp/prompt.txt
    
    # Send request
    START_TIME=$(date +%s.%N)
    
    curl -s -X POST http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\": \"meta-llama/Llama-3.2-1B-Instruct\", \"prompt\": $(cat /tmp/prompt.txt | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))'), \"max_tokens\": $MAX_TOKENS, \"temperature\": 0}" \
      > /dev/null
    
    END_TIME=$(date +%s.%N)
    DURATION=$(echo "$END_TIME - START_TIME" | bc)
    
    # Wait for offload
    sleep 5
    
    # VRAM after
    VRAM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    
    # Disk usage
    DISK_SIZE=$(du -sh /tmp/lmcache/lmcache_disk/ 2>/dev/null | cut -f1)
    
    # Save result
    cat > "$RESULTS_DIR/${MODE}_${PREFILL}.json" << EOF
{
  "mode": "$MODE",
  "prefill": $PREFILL,
  "max_tokens": $MAX_TOKENS,
  "duration_sec": $DURATION,
  "vram_before_mb": $VRAM_BEFORE,
  "vram_after_mb": $VRAM_AFTER,
  "vram_delta_mb": $((VRAM_AFTER - VRAM_BEFORE)),
  "disk_size": "$DISK_SIZE"
}
EOF
    
    echo "Result saved: ${MODE}_${PREFILL}.json"
    echo "VRAM Delta: $((VRAM_BEFORE - VRAM_AFTER)) MB"
    echo "Disk Size: $DISK_SIZE"
}

# Start with Native mode
echo ""
echo "========================================"
echo "Phase 1: Native Mode (torch serialization)"
echo "========================================"

./start_vllm_native.sh

# Warmup
echo "Warming up..."
sleep 30

# Run tests for each prefill size
for PREFILL in "${PREFILL_SIZES[@]}"; do
    run_test "native" "$PREFILL"
    sleep $COOLDOWN
done

# Stop vLLM
echo ""
echo "Stopping vLLM..."
./stop_vllm.sh
sleep 10

# Switch to CacheGen mode
echo ""
echo "========================================"
echo "Phase 2: CacheGen Mode (compression)"
echo "========================================"

./start_vllm_cachegen.sh

# Warmup
echo "Warming up..."
sleep 30

# Run tests for each prefill size
for PREFILL in "${PREFILL_SIZES[@]}"; do
    run_test "cachegen" "$PREFILL"
    sleep $COOLDOWN
done

# Stop vLLM
echo ""
echo "Stopping vLLM..."
./stop_vllm.sh

# Generate summary
echo ""
echo "========================================"
echo "Experiment Complete!"
echo "========================================"
echo "Results saved in: $RESULTS_DIR/"
echo ""

# Print summary
echo "Summary:"
echo "|---------|--------|----------|----------|----------|----------|"
echo "| Mode    | Prefill | Duration | VRAM Δ   | Disk     |"
echo "|---------|--------|----------|----------|----------|"

for MODE in native cachegen; do
    for PREFILL in "${PREFILL_SIZES[@]}"; do
        if [ -f "$RESULTS_DIR/${MODE}_${PREFILL}.json" ]; then
            DUR=$(cat "$RESULTS_DIR/${MODE}_${PREFILL}.json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['duration_sec'])")
            VRAM=$(cat "$RESULTS_DIR/${MODE}_${PREFILL}.json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['vram_delta_mb'])")
            DISK=$(cat "$RESULTS_DIR/${MODE}_${PREFILL}.json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['disk_size'])")
            printf "| %-7s | %7s | %8s | %8s | %8s |\n" "$MODE" "$PREFILL" "${DUR}s" "${VRAM}MB" "$DISK"
        fi
    done
done

echo "|---------|--------|----------|----------|----------|"
