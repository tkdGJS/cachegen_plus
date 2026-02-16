#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/noslab-gpu/tkdgjs/tkdgjs/lib/python3.10/site-packages')

import os
os.environ["LMCACHE_VRAM_LOG"] = "1"

import torch
import json
import subprocess
import time

PORT = 8000
MODEL = "meta-llama/Llama-3.2-1B-Instruct"

def cleanup_gpu():
    subprocess.run(["pkill", "-9", "-f", "vllm"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", "EngineCore"], stderr=subprocess.DEVNULL)
    time.sleep(3)

def get_vram_torch():
    return torch.cuda.memory_allocated() / 1024**3

def get_vram_peak_torch():
    return torch.cuda.max_memory_allocated() / 1024**3

def get_nvidia_smi():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0:
        return float(result.stdout.strip().split('\n')[0]) / 1024
    return 0.0

def test_native():
    print("\n" + "="*60)
    print("Testing Native Mode (KV Cache Copy Only)")
    print("="*60)
    
    torch.cuda.reset_peak_memory_stats()
    
    num_layers = 16
    num_tokens = 4096
    num_heads = 32
    head_size = 128
    
    print(f"Creating KV cache: {num_layers}x2x{num_tokens}x{num_heads}x{head_size}")
    kv_cache = torch.randn(num_layers, 2, num_tokens, num_heads, head_size, 
                          dtype=torch.float16, device='cuda')
    
    mem_kv = get_vram_torch()
    print(f"After KV cache allocation: {mem_kv:.4f} GB")
    
    torch.cuda.reset_peak_memory_stats()
    mem_before = get_vram_torch()
    
    print("Performing KV cache clone (copy)...")
    kv_clone = kv_cache.clone()
    torch.cuda.synchronize()
    
    mem_after = get_vram_torch()
    mem_peak = get_vram_peak_torch()
    
    print(f"Memory after clone: {mem_after:.4f} GB")
    print(f"Peak memory: {mem_peak:.4f} GB")
    print(f"Memory increase: {mem_after - mem_before:.4f} GB")
    print(f"Peak increase: {mem_peak - mem_before:.4f} GB")
    
    nvidia_vram = get_nvidia_smi()
    print(f"nvidia-smi VRAM: {nvidia_vram:.2f} GB")
    
    del kv_cache, kv_clone
    torch.cuda.empty_cache()
    
    return {
        "mode": "native",
        "kv_cache_gb": mem_kv,
        "mem_after_clone_gb": mem_after,
        "peak_during_clone_gb": mem_peak,
        "memory_increase_gb": mem_after - mem_before,
        "peak_increase_gb": mem_peak - mem_before,
        "nvidia_smi_gb": nvidia_vram
    }

def test_cachegen():
    print("\n" + "="*60)
    print("Testing CacheGen Mode (Compression)")
    print("="*60)
    
    from lmcache.storage_backend.serde.cachegen_encoder import encode_function
    from lmcache.storage_backend.serde.cachegen_basics import CacheGenConfig
    
    torch.cuda.reset_peak_memory_stats()
    
    num_layers = 16
    num_tokens = 4096
    num_heads = 32
    head_size = 128
    
    print(f"Creating KV cache: {num_layers}x2x{num_tokens}x{num_heads}x{head_size}")
    kv_cache = torch.randn(num_layers, 2, num_tokens, num_heads, head_size,
                          dtype=torch.float16, device='cuda')
    
    mem_kv = get_vram_torch()
    print(f"After KV cache allocation: {mem_kv:.4f} GB")
    
    torch.cuda.reset_peak_memory_stats()
    mem_before = get_vram_torch()
    
    print("Running CacheGen compression...")
    config = CacheGenConfig.from_model_name("meta-llama/Llama-3.2-1B-Instruct")
    key_bins = torch.tensor([32] * num_layers, device='cuda')
    value_bins = torch.tensor([32] * num_layers, device='cuda')
    
    result = encode_function(kv_cache, config, key_bins, value_bins, num_tokens)
    torch.cuda.synchronize()
    
    mem_after = get_vram_torch()
    mem_peak = get_vram_peak_torch()
    
    print(f"Memory after compression: {mem_after:.4f} GB")
    print(f"Peak memory: {mem_peak:.4f} GB")
    print(f"Memory increase: {mem_after - mem_before:.4f} GB")
    print(f"Peak increase: {mem_peak - mem_before:.4f} GB")
    
    nvidia_vram = get_nvidia_smi()
    print(f"nvidia-smi VRAM: {nvidia_vram:.2f} GB")
    
    del kv_cache, result
    torch.cuda.empty_cache()
    
    return {
        "mode": "cachegen",
        "kv_cache_gb": mem_kv,
        "mem_after_compress_gb": mem_after,
        "peak_during_compress_gb": mem_peak,
        "memory_increase_gb": mem_after - mem_before,
        "peak_increase_gb": mem_peak - mem_before,
        "nvidia_smi_gb": nvidia_vram
    }

def main():
    cleanup_gpu()
    torch.cuda.set_device(0)
    
    print("Initial VRAM:", get_vram_torch(), "GB")
    print("Initial nvidia-smi:", get_nvidia_smi(), "GB")
    
    native_result = test_native()
    
    cleanup_gpu()
    time.sleep(2)
    torch.cuda.reset_peak_memory_stats()
    
    cachegen_result = test_cachegen()
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"\n{'Metric':<30} {'Native':<15} {'CacheGen':<15} {'Diff':<12}")
    print("-" * 75)
    print(f"{'KV Cache Size':<30} {native_result['kv_cache_gb']:<15.4f} {cachegen_result['kv_cache_gb']:<15.4f} {'N/A':<12}")
    print(f"{'Memory After Operation':<30} {native_result['mem_after_clone_gb']:<15.4f} {cachegen_result['mem_after_compress_gb']:<15.4f} {cachegen_result['mem_after_compress_gb'] - native_result['mem_after_clone_gb']:+<12.4f}")
    print(f"{'Peak During Operation':<30} {native_result['peak_during_clone_gb']:<15.4f} {cachegen_result['peak_during_compress_gb']:<15.4f} {cachegen_result['peak_during_compress_gb'] - native_result['peak_during_clone_gb']:+<12.4f}")
    print(f"{'Memory Increase':<30} {native_result['memory_increase_gb']:<15.4f} {cachegen_result['memory_increase_gb']:<15.4f} {cachegen_result['memory_increase_gb'] - native_result['memory_increase_gb']:+<12.4f}")
    print(f"{'Peak Increase':<30} {native_result['peak_increase_gb']:<15.4f} {cachegen_result['peak_increase_gb']:<15.4f} {cachegen_result['peak_increase_gb'] - native_result['peak_increase_gb']:+<12.4f}")
    
    results = {
        "native": native_result,
        "cachegen": cachegen_result
    }
    
    with open("/home/noslab-gpu/tkdgjs/experiment/result/vram_measured_data.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to: /home/noslab-gpu/tkdgjs/experiment/result/vram_measured_data.json")

if __name__ == "__main__":
    main()
