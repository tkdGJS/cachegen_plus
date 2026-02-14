#!/usr/bin/env python3
"""
VRAM Timeline Test - Track VRAM usage over time during compression
"""
import sys
sys.path.insert(0, '/home/noslab-gpu/tkdgjs/tkdgjs/lib/python3.10/site-packages')

import torch
import time
from lmcache.storage_backend.serde.cachegen_encoder import encode_function
from lmcache.storage_backend.serde.cachegen_basics import CacheGenConfig

# Config
num_layers = 16
num_tokens = 4096
num_heads = 32
head_size = 128

print("="*70)
print("VRAM Timeline - Compression over Time")
print("="*70)

# Create KV cache
kv_cache = torch.randn(num_layers, 2, num_tokens, num_heads, head_size, 
                       dtype=torch.float16, device='cuda')
print(f"\nKV cache: {num_layers} layers, {num_tokens} tokens")

# Baseline
torch.cuda.reset_peak_memory_stats()
base_mem = torch.cuda.memory_allocated() / 1024**3

# Timeline storage
timeline = []
start_time = time.time()

# Function to record timeline
def record_step(name):
    elapsed = time.time() - start_time
    mem = torch.cuda.memory_allocated() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    timeline.append({
        'time': elapsed,
        'name': name,
        'mem_gb': mem,
        'peak_gb': peak,
        'mem_delta': mem - base_mem,
        'peak_delta': peak - base_mem
    })

# Custom encoding with timeline
config = CacheGenConfig.from_model_name("meta-llama/Llama-3.2-1B-Instruct")
key_bins = torch.tensor([32] * num_layers, device='cuda')
value_bins = torch.tensor([32] * num_layers, device='cuda')

# Step-by-step with timeline
print("\n--- Encoding Steps ---\n")

record_step("START")
print(f"[{timeline[-1]['time']*1000:8.2f}ms] START: {timeline[-1]['mem_gb']:.4f} GB")

# split_kv
from lmcache.storage_backend.serde.cachegen_encoder import _split_kv
fp_k, fp_v = _split_kv(kv_cache)
nchannels = fp_k.shape[-1]
nlayers = fp_k.shape[0] + fp_v.shape[0]
record_step("01_split_kv")
print(f"[{timeline[-1]['time']*1000:8.2f}ms] split_kv: {timeline[-1]['mem_gb']:.4f} GB (+{timeline[-1]['mem_delta']:.4f})")

# quant_key
from lmcache.storage_backend.serde.cachegen_encoder import torch_quant_vectorized
new_key, max_tensors_key = torch_quant_vectorized(key_bins, fp_k)
record_step("02_quant_key")
print(f"[{timeline[-1]['time']*1000:8.2f}ms] quant_key: {timeline[-1]['mem_gb']:.4f} GB (+{timeline[-1]['mem_delta']:.4f})")

# quant_value
new_value, max_tensors_value = torch_quant_vectorized(value_bins, fp_v)
record_step("03_quant_value")
print(f"[{timeline[-1]['time']*1000:8.2f}ms] quant_value: {timeline[-1]['mem_gb']:.4f} GB (+{timeline[-1]['mem_delta']:.4f})")

# cat
encode_input = torch.cat((new_key, new_value), dim=0).reshape(nlayers, num_tokens, nchannels)
record_step("04_cat_encode_input")
print(f"[{timeline[-1]['time']*1000:8.2f}ms] cat_encode_input: {timeline[-1]['mem_gb']:.4f} GB (+{timeline[-1]['mem_delta']:.4f})")

# calculate_cdf
from lmcache.storage_backend.serde import cachegen_encoder
new_cdf_key = cachegen_encoder.lmc_ops.calculate_cdf(new_key, int(key_bins.max()))
new_cdf_value = cachegen_encoder.lmc_ops.calculate_cdf(new_value, int(value_bins.max()))
cdf_int = torch.cat([new_cdf_key, new_cdf_value])
record_step("05_calculate_cdf")
print(f"[{timeline[-1]['time']*1000:8.2f}ms] calculate_cdf: {timeline[-1]['mem_gb']:.4f} GB (+{timeline[-1]['mem_delta']:.4f})")

# output_buffer
from lmcache.storage_backend.serde import cachegen_basics
output_buffer = torch.zeros(
    (nlayers, nchannels, cachegen_basics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK),
    dtype=torch.uint8, device='cuda'
)
output_lengths = torch.zeros((nlayers, nchannels), dtype=torch.int32, device='cuda')
record_step("06_output_buffer")
print(f"[{timeline[-1]['time']*1000:8.2f}ms] output_buffer: {timeline[-1]['mem_gb']:.4f} GB (+{timeline[-1]['mem_delta']:.4f})")

# encode_ntokens
from lmcache.storage_backend.serde.cachegen_encoder import encode_ntokens
for i in range(0, num_tokens, cachegen_basics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK):
    end = min(i + cachegen_basics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK, num_tokens)
    bytestream = encode_ntokens(cdf_int, encode_input[:, i:end, :], output_buffer, output_lengths)
    if i == 0:
        record_step("07_encode_ntokens")
        print(f"[{timeline[-1]['time']*1000:8.2f}ms] encode_ntokens: {timeline[-1]['mem_gb']:.4f} GB (+{timeline[-1]['mem_delta']:.4f})")

# Final
torch.cuda.synchronize()
record_step("END")
print(f"[{timeline[-1]['time']*1000:8.2f}ms] END: {timeline[-1]['mem_gb']:.4f} GB (+{timeline[-1]['mem_delta']:.4f})")

# Summary
print("\n" + "="*70)
print("VRAM Timeline Summary")
print("="*70)
print(f"\n{'Time (ms)':<12} {'Step':<25} {'VRAM (GB)':<12} {'Delta (GB)':<12}")
print("-"*65)
for t in timeline:
    print(f"{t['time']*1000:<12.2f} {t['name']:<25} {t['mem_gb']:<12.4f} {t['mem_delta']:+.4f}")

# Visualization
print("\n" + "="*70)
print("VRAM Usage Over Time (ASCII Timeline)")
print("="*70)

max_mem = max(t['mem_gb'] for t in timeline)
min_mem = min(t['mem_gb'] for t in timeline)
mem_range = max_mem - min_mem

for t in timeline:
    if mem_range > 0:
        bar_len = int((t['mem_gb'] - min_mem) / mem_range * 50)
    else:
        bar_len = 0
    bar = "█" * bar_len + "░" * (50 - bar_len)
    print(f"{t['time']*1000:7.2f}ms |{bar}| {t['mem_gb']:.3f} GB")

print("\n" + "="*70)
print("Peak VRAM: {:.4f} GB (delta: {:.4f} GB)".format(
    max(t['peak_gb'] for t in timeline),
    max(t['peak_gb'] for t in timeline) - base_mem
))
print("="*70)
