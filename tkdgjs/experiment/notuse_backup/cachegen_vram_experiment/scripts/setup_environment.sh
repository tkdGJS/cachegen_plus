#!/bin/bash
#===============================================================================
# CacheGen VRAM Experiment - Environment Setup Script
#
# This script prepares the environment for running CacheGen VRAM experiments.
# Run this first before starting vLLM.
#
# Usage: ./setup_environment.sh
#===============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "CacheGen VRAM Experiment - Setup"
echo "========================================"

# 1. Check GPU
echo "[1/5] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    if [ "$GPU_FREE" -lt 10000 ]; then
        echo "WARNING: Less than 10GB VRAM available. Close other GPU processes."
    fi
else
    echo "ERROR: nvidia-smi not found"
    exit 1
fi

# 2. Check ports
echo "[2/5] Checking ports..."
for PORT in 8000 6999; do
    if lsof -i :$PORT &> /dev/null; then
        echo "WARNING: Port $PORT is in use. Kill existing process before continuing."
    else
        echo "  Port $PORT: OK"
    fi
done

# 3. Create directories
echo "[3/5] Creating directories..."
mkdir -p "$PROJECT_DIR/logs"
mkdir -p /tmp/lmcache/lmcache_disk
mkdir -p /tmp/lmcache/lmcache_native_disk
mkdir -p /tmp/lmcache/lmcache_cachegen_disk
echo "  Created: $PROJECT_DIR/logs"
echo "  Created: /tmp/lmcache/lmcache_disk"

# 4. Clear previous disk cache
echo "[4/5] Clearing disk cache..."
rm -rf /tmp/lmcache/lmcache_disk/*
echo "  Disk cache cleared"

# 5. Set environment variables
echo "[5/5] Environment variables..."
export PATH=/home/noslab-gpu/tkdgjs/tkdgjs/bin:$PATH
export PYTHONHASHSEED=0
echo "  PATH set"
echo "  PYTHONHASHSEED=0"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Start vLLM with Native mode:   ./start_vllm_native.sh"
echo "  2. Start vLLM with CacheGen mode: ./start_vllm_cachegen.sh"
echo "  3. Run tests:                      ./test_vram.sh"
echo ""
