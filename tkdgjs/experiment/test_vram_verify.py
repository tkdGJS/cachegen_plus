#!/usr/bin/env python3
"""VRAM Layout Verification - Direct Test"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_sweep_experiment import FullVRAMMonitor, VRAMSnapshot
import json

def test_vram_layout():
    print("="*60)
    print("VRAM Layout Verification - Direct Test")
    print("="*60)
    
    # Test 1: vLLM not running (idle GPU)
    print("\n[Test 1] GPU Idle (vLLM not running)")
    monitor = FullVRAMMonitor(port=8000, is_cachegen=False)
    snapshot = monitor.measure(is_cachegen=False)
    snapshot.print_layout("GPU Idle")
    
    print(f"  Sum match: {abs(snapshot.sum_diff_gb) < 0.01}")
    
    # Test 2: Simulate vLLM metrics
    print("\n[Test 2] Simulated vLLM Metrics")
    
    # Create a mock metrics response
    test_metrics = """
# HELP gpu_memory_utilization GPU memory utilization
# TYPE gpu_memory_utilization gauge
gpu_memory_utilization="0.7"
# HELP num_gpu_blocks Number of GPU blocks
# TYPE num_gpu_blocks gauge
num_gpu_blocks="512"
# HELP num_gpu_blocks_free Number of free GPU blocks  
# TYPE num_gpu_blocks_free gauge
num_gpu_blocks_free="256"
# HELP block_size KV cache block size
# TYPE block_size gauge
block_size="16"
# HELP vllm:kv_cache_usage_perc KV cache usage percentage
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{} 50.0
"""
    
    import requests
    from unittest.mock import patch, MagicMock
    
    # Mock the metrics endpoint
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = test_metrics
    
    with patch.object(requests, 'get', return_value=mock_response):
        # Test with vLLM running
        monitor2 = FullVRAMMonitor(port=8000, is_cachegen=False)
        
        # Set baseline first
        baseline = monitor2.set_baseline()
        print(f"  Baseline set: used_vram={baseline.used_vram_gb:.2f}GB")
        
        # Now measure with simulated vLLM metrics
        snapshot2 = monitor2.measure(baseline=baseline, is_cachegen=False)
        snapshot2.print_layout("vLLM Running (Simulated)")
        
        print(f"\n  VRAM Layout Breakdown:")
        print(f"    Model Weights:     {snapshot2.model_weights_gb:.4f} GB")
        print(f"    KV Allocated:      {snapshot2.vllm_kv_cache_allocated_gb:.4f} GB")
        print(f"    KV Used:           {snapshot2.vllm_kv_cache_used_gb:.4f} GB")
        print(f"    Activation:        {snapshot2.activation_tensors_gb:.4f} GB")
        print(f"    CUDA Runtime:      {snapshot2.cuda_runtime_gb:.4f} GB")
        print(f"    Sum:               {snapshot2.sum_validated_gb:.4f} GB")
        print(f"    nvidia-smi used:   {snapshot2.used_vram_gb:.4f} GB")
        print(f"    Diff:              {snapshot2.sum_diff_gb:+.4f} GB")
        
        sum_matches = abs(snapshot2.sum_diff_gb) < 0.5
        print(f"\n  Sum match (<0.5GB): {sum_matches}")
    
    # Test 3: CacheGen mode
    print("\n[Test 3] CacheGen Mode (Simulated)")
    with patch.object(requests, 'get', return_value=mock_response):
        monitor3 = FullVRAMMonitor(port=8000, is_cachegen=True)
        
        baseline3 = monitor3.set_baseline()
        
        # Simulate VRAM increase from baseline
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5
        )
        current_used = float(result.stdout.strip()) / 1024
        
        # Manually simulate the case where baseline had less VRAM
        baseline3.used_vram_gb = current_used - 0.5  # Simulate 0.5GB increase
        
        snapshot3 = monitor3.measure(baseline=baseline3, is_cachegen=True)
        snapshot3.print_layout("CacheGen Mode")
        
        print(f"\n  CacheGen VRAM:")
        print(f"    Encoder:           {snapshot3.cachegen_encoder_gb:.4f} GB")
        print(f"    Decoder:           {snapshot3.cachegen_decoder_gb:.4f} GB")
        print(f"    Compressed KV:     {snapshot3.cachegen_compressed_kv_gb:.4f} GB")
        print(f"    Total:             {snapshot3.cachegen_total_gb:.4f} GB")
    
    print("\n" + "="*60)
    print("Verification Complete")
    print("="*60)

if __name__ == "__main__":
    test_vram_layout()
