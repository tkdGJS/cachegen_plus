#!/usr/bin/env python3
"""
Direct test of VRAM measurement in CacheGen encoder
without needing full vLLM startup
"""
import sys
sys.path.insert(0, '/home/noslab-gpu/tkdgjs/tkdgjs/lib/python3.10/site-packages')

# Enable VRAM logging
import os
os.environ["LMCACHE_VRAM_LOG"] = "1"

import torch
from lmcache.storage_backend.serde.cachegen_encoder import encode_function
from lmcache.storage_backend.serde.cachegen_basics import CacheGenConfig, CacheGenGPUEncoderOutput

# Create test KV cache (simulating Llama-3.2-1B)
# Shape: [num_layers, 2 (K,V), num_tokens, num_heads, head_size]
num_layers = 16
num_tokens = 4096
num_heads = 32
head_size = 128

print("="*60)
print("Testing VRAM measurement in CacheGen encode_function")
print("="*60)

print(f"\nCreating test KV cache...")
print(f"  Layers: {num_layers}")
print(f"  Tokens: {num_tokens}")
print(f"  Heads: {num_heads}")
print(f"  Head size: {head_size}")

# Create KV cache tensor
kv_cache = torch.randn(num_layers, 2, num_tokens, num_heads, head_size, dtype=torch.float16, device='cuda')
kv_size_gb = kv_cache.element_size() * kv_cache.numel() / 1024**3
print(f"  KV cache size: {kv_size_gb:.4f} GB")

# Get baseline VRAM
torch.cuda.reset_peak_memory_stats()
mem_before = torch.cuda.memory_allocated() / 1024**3
print(f"\nVRAM before encoding: {mem_before:.4f} GB")

# Create config
config = CacheGenConfig.from_model_name("meta-llama/Llama-3.2-1B-Instruct")
key_bins = torch.tensor([32] * num_layers, device='cuda')
value_bins = torch.tensor([32] * num_layers, device='cuda')

# Run encoding
print("\nRunning encode_function...")
result = encode_function(kv_cache, config, key_bins, value_bins, num_tokens)

# Get VRAM after encoding
mem_after = torch.cuda.memory_allocated() / 1024**3
peak_mem = torch.cuda.max_memory_allocated() / 1024**3

print(f"\nVRAM after encoding:  {mem_after:.4f} GB")
print(f"Peak VRAM:            {peak_mem:.4f} GB")
print(f"Memory increase:      {mem_after - mem_before:.4f} GB")
print(f"Peak increase:        {peak_mem - mem_before:.4f} GB")

# Now test WITHOUT compression (just copy)
print("\n" + "="*60)
print("Testing WITHOUT compression (simple copy)")
print("="*60)

torch.cuda.reset_peak_memory_stats()
mem_before2 = torch.cuda.memory_allocated() / 1024**3
print(f"\nVRAM before copy: {mem_before2:.4f} GB")

# Simple copy (simulating Native mode - no compression)
copy_result = kv_cache.clone()
torch.cuda.synchronize()

mem_after2 = torch.cuda.memory_allocated() / 1024**3
peak_mem2 = torch.cuda.max_memory_allocated() / 1024**3

print(f"VRAM after copy:   {mem_after2:.4f} GB")
print(f"Peak VRAM:         {peak_mem2:.4f} GB")
print(f"Memory increase:    {mem_after2 - mem_before2:.4f} GB")
print(f"Peak increase:     {peak_mem2 - mem_before2:.4f} GB")

# Comparison
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"\n{'Mode':<20} {'Mem Increase':<15} {'Peak Increase':<15}")
print("-"*55)
print(f"{'CacheGen (compress)':<20} {mem_after - mem_before:<15.4f} {peak_mem - mem_before:<15.4f}")
print(f"{'Native (copy only)':<20} {mem_after2 - mem_before2:<15.4f} {peak_mem2 - mem_before2:<15.4f}")

diff = (mem_after - mem_before) - (mem_after2 - mem_before2)
peak_diff = (peak_mem - mem_before) - (peak_mem2 - mem_before2)

print(f"\nDifference (CacheGen - Native):")
print(f"  Memory increase: {diff:+.4f} GB")
print(f"  Peak increase:   {peak_diff:+.4f} GB")

if diff > 0:
    print(f"\nRESULT: CacheGen uses MORE VRAM than Native (+{diff:.4f} GB)")
else:
    print(f"\nRESULT: CacheGen uses LESS or SAME VRAM as Native ({diff:.4f} GB)")
