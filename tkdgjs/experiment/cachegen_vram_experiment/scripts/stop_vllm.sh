#!/bin/bash
#===============================================================================
# Stop vLLM instance
#
# Usage: ./stop_vllm.sh
#===============================================================================

echo "Stopping vLLM..."

# Kill vLLM process
pkill -f "vllm serve" || echo "No vLLM process found"

# Wait for cleanup
sleep 3

# Check VRAM
echo ""
echo "VRAM Status:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits

echo ""
echo "vLLM stopped."
