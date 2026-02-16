#!/usr/bin/env python3
"""
VRAM Timeline - Synchronized Polling + Event Markers
The key is to synchronize the polling with the compression operation
"""
import sys
sys.path.insert(0, '/home/noslab-gpu/tkdgjs/tkdgjs/lib/python3.10/site-packages')

import torch
import time
from collections import defaultdict

# Timeline storage
timeline = []

def record(type_name, name=""):
    """Record a point in timeline"""
    mem = torch.cuda.memory_allocated() / 1024**3
    timeline.append({
        'type': type_name,
        'name': name,
        'time': time.time(),
        'mem_gb': mem
    })

# Test 1: With event markers only (synchronized)
print("="*70)
print("Test 1: Event Markers (Synchronized with Operations)")
print("="*70)

torch.cuda.reset_peak_memory_stats()
timeline.clear()
base_mem = torch.cuda.memory_allocated() / 1024**3
start_time = time.time()

record("start", "baseline")

# Create KV cache
num_layers = 16
num_tokens = 4096
num_heads = 32
head_size = 128
kv_cache = torch.randn(num_layers, 2, num_tokens, num_heads, head_size, 
                       dtype=torch.float16, device='cuda')

record("01", "kv_allocated")

# Import encoder functions
from lmcache.storage_backend.serde.cachegen_encoder import (
    _split_kv, torch_quant_vectorized, encode_ntokens
)
from lmcache.storage_backend.serde import cachegen_basics
import lmcache.storage_backend.serde.cachegen_encoder as ce

# Create config
config = cachegen_basics.CacheGenConfig.from_model_name("meta-llama/Llama-3.2-1B-Instruct")
key_bins = torch.tensor([32] * num_layers, device='cuda')
value_bins = torch.tensor([32] * num_layers, device='cuda')

# Compression with event markers
record("02", "start_compression")

fp_k, fp_v = _split_kv(kv_cache)
record("03", "split_kv")

new_key, max_tensors_key = torch_quant_vectorized(key_bins, fp_k)
record("04", "quant_key")

new_value, max_tensors_value = torch_quant_vectorized(value_bins, fp_v)
record("05", "quant_value")

nchannels = fp_k.shape[-1]
nlayers = fp_k.shape[0] + fp_v.shape[0]
encode_input = torch.cat((new_key, new_value), dim=0).reshape(nlayers, num_tokens, nchannels)
record("06", "cat_encode_input")

new_cdf_key = ce.lmc_ops.calculate_cdf(new_key, int(key_bins.max()))
new_cdf_value = ce.lmc_ops.calculate_cdf(new_value, int(value_bins.max()))
cdf_int = torch.cat([new_cdf_key, new_cdf_value])
record("07", "calculate_cdf")

output_buffer = torch.zeros(
    (nlayers, nchannels, cachegen_basics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK),
    dtype=torch.uint8, device='cuda'
)
output_lengths = torch.zeros((nlayers, nchannels), dtype=torch.int32, device='cuda')
record("08", "output_buffer")

for i in range(0, num_tokens, cachegen_basics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK):
    end = min(i + cachegen_basics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK, num_tokens)
    bytestream = encode_ntokens(cdf_int, encode_input[:, i:end, :], output_buffer, output_lengths)
    if i == 0:
        record("09", "encode_ntokens")

torch.cuda.synchronize()
record("10", "complete")

end_time = time.time()

# Calculate deltas
base_time = timeline[0]['time']
base_mem = timeline[0]['mem_gb']

print(f"\n{'Time (ms)':<12} {'Step':<25} {'VRAM (GB)':<12} {'Delta (GB)':<12}")
print("-"*65)
for t in timeline:
    delta_time = (t['time'] - base_time) * 1000
    delta_mem = t['mem_gb'] - base_mem
    print(f"{delta_time:<12.2f} {t['name']:<25} {t['mem_gb']:<12.4f} {delta_mem:+.4f}")

print(f"\nTotal duration: {(end_time - start_time)*1000:.2f}ms")

# Test 2: Continuous polling with triggering
print("\n" + "="*70)
print("Test 2: Continuous Polling (Triggered by Operation)")
print("="*70)

torch.cuda.reset_peak_memory_stats()
timeline.clear()
base_mem = torch.cuda.memory_allocated() / 1024**3
start_time = time.time()

# Create KV cache
kv_cache2 = torch.randn(num_layers, 2, num_tokens, num_heads, head_size, 
                        dtype=torch.float16, device='cuda')

# Start polling in a tight loop during compression
import threading

poll_data = []
poll_active = True

def poll_continuously():
    """Poll VRAM as fast as possible"""
    while poll_active:
        mem = torch.cuda.memory_allocated() / 1024**3
        poll_data.append({
            'time': time.time(),
            'mem_gb': mem
        })
        # No sleep - maximum polling rate

# Start polling
poll_thread = threading.Thread(target=poll_continuously)
poll_thread.start()

# Small delay to let polling start
time.sleep(0.001)

# Run compression (trigger)
record("trigger_start", "compression_start")

# Do the actual compression work
fp_k2, fp_v2 = _split_kv(kv_cache2)
new_key2, _ = torch_quant_vectorized(key_bins, fp_k2)
new_value2, _ = torch_quant_vectorized(value_bins, fp_v2)
encode_input2 = torch.cat((new_key2, new_value2), dim=0).reshape(nlayers, num_tokens, nchannels)
new_cdf_key2 = ce.lmc_ops.calculate_cdf(new_key2, int(key_bins.max()))
new_cdf_value2 = ce.lmc_ops.calculate_cdf(new_value2, int(value_bins.max()))
cdf_int2 = torch.cat([new_cdf_key2, new_cdf_value2])

record("trigger_end", "compression_end")

# Stop polling
time.sleep(0.001)
poll_active = False
poll_thread.join()

end_time = time.time()

# Find VRAM spike
if poll_data:
    poll_start = poll_data[0]['time']
    poll_base = poll_data[0]['mem_gb']
    
    print(f"\nPolling samples: {len(poll_data)}")
    
    # Show timeline around the spike
    print(f"\n{'Time (ms)':<12} {'VRAM (GB)':<12} {'Delta (GB)':<12}")
    print("-"*40)
    
    # Find significant changes
    for i, p in enumerate(poll_data[:100]):  # First 100
        delta = (p['time'] - poll_start) * 1000
        delta_mem = p['mem_gb'] - poll_base
        print(f"{delta:<12.2f} {p['mem_gb']:<12.4f} {delta_mem:+.4f}")
    
    # Find max
    max_mem = max(p['mem_gb'] for p in poll_data)
    min_mem = min(p['mem_gb'] for p in poll_data)
    print(f"\nMax VRAM: {max_mem:.4f} GB")
    print(f"Min VRAM: {min_mem:.4f} GB")
    print(f"Spike: {max_mem - min_mem:.4f} GB")

print("\n" + "="*70)
print("Summary: Combined VRAM Timeline")
print("="*70)
print("""
Approach:
1. Event Markers: Record VRAM at each operation step (synchronized)
2. Continuous Polling: Background polling to catch transient spikes

The event markers show WHERE the VRAM increase happens.
The polling shows WHEN it happens in continuous time.
""")
