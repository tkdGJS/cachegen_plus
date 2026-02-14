#!/usr/bin/env python3
"""
VRAM Timeline with Hybrid Polling + Event Markers
- Continuous polling at 0.01ms intervals
- Event markers from CacheGen operations
- Combined timeline
"""
import sys
sys.path.insert(0, '/home/noslab-gpu/tkdgjs/tkdgjs/lib/python3.10/site-packages')

import torch
import time
import threading
from collections import defaultdict

# Global storage for timeline
vram_timeline = []
vram_lock = threading.Lock()
polling_active = True

def poll_vram(interval=0.0001):  # 0.1ms = 100 microseconds
    """Background polling at high frequency"""
    base_mem = torch.cuda.memory_allocated() / 1024**3
    start_time = time.time()
    
    while polling_active:
        mem = torch.cuda.memory_allocated() / 1024**3
        elapsed = time.time() - start_time
        
        with vram_lock:
            vram_timeline.append({
                'type': 'poll',
                'time': elapsed,
                'mem_gb': mem,
                'mem_delta': mem - base_mem
            })
        
        time.sleep(interval)

def add_event_marker(name):
    """Add event marker from CacheGen operations"""
    mem = torch.cuda.memory_allocated() / 1024**3
    base_mem = vram_timeline[0]['mem_gb'] if vram_timeline else 0
    
    with vram_lock:
        vram_timeline.append({
            'type': 'event',
            'name': name,
            'time': time.time() - start_time,
            'mem_gb': mem,
            'mem_delta': mem - base_mem
        })

# Start time for reference
start_time = time.time()

# Modified encode_function with event markers
def encode_with_timeline(kv, config, key_bins, value_bins, chunk_size):
    """encode_function with event markers"""
    from lmcache.storage_backend.serde.cachegen_encoder import (
        _split_kv, torch_quant_vectorized, encode_ntokens
    )
    from lmcache.storage_backend.serde import cachegen_basics
    
    add_event_marker("01_input_kv")
    
    # split_kv
    fp_k, fp_v = _split_kv(kv)
    nchannels = fp_k.shape[-1]
    nlayers = fp_k.shape[0] + fp_v.shape[0]
    add_event_marker("02_split_kv")
    
    # quant_key
    new_key, max_tensors_key = torch_quant_vectorized(key_bins, fp_k)
    add_event_marker("03_quant_key")
    
    # quant_value
    new_value, max_tensors_value = torch_quant_vectorized(value_bins, fp_v)
    add_event_marker("04_quant_value")
    
    # cat
    encode_input = torch.cat((new_key, new_value), dim=0).reshape(nlayers, chunk_size, nchannels)
    add_event_marker("05_cat_encode_input")
    
    # calculate_cdf
    import lmcache.storage_backend.serde.cachegen_encoder as ce
    new_cdf_key = ce.lmc_ops.calculate_cdf(new_key, int(key_bins.max()))
    new_cdf_value = ce.lmc_ops.calculate_cdf(new_value, int(value_bins.max()))
    cdf_int = torch.cat([new_cdf_key, new_cdf_value])
    add_event_marker("06_calculate_cdf")
    
    # output_buffer
    output_buffer = torch.zeros(
        (nlayers, nchannels, cachegen_basics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK),
        dtype=torch.uint8, device='cuda'
    )
    output_lengths = torch.zeros((nlayers, nchannels), dtype=torch.int32, device='cuda')
    add_event_marker("07_output_buffer")
    
    # encode_ntokens
    for i in range(0, chunk_size, cachegen_basics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK):
        end = min(i + cachegen_basics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK, chunk_size)
        bytestream = encode_ntokens(cdf_int, encode_input[:, i:end, :], output_buffer, output_lengths)
        if i == 0:
            add_event_marker("08_encode_ntokens")
    
    add_event_marker("09_complete")
    torch.cuda.synchronize()
    
    return cdf_int  # Simplified return

# Main test
print("="*70)
print("VRAM Timeline - Hybrid Polling + Event Markers")
print("="*70)

# Create KV cache
num_layers = 16
num_tokens = 4096
num_heads = 32
head_size = 128

kv_cache = torch.randn(num_layers, 2, num_tokens, num_heads, head_size, 
                       dtype=torch.float16, device='cuda')
print(f"\nKV cache: {num_layers} layers, {num_tokens} tokens")

# Reset
torch.cuda.reset_peak_memory_stats()
base_mem = torch.cuda.memory_allocated() / 1024**3
start_time = time.time()
vram_timeline.clear()

# Start polling thread
polling_active = True
poll_thread = threading.Thread(target=poll_vram, args=(0.0001,))  # 0.1ms
poll_thread.start()

print("\n--- Starting compression with continuous polling ---\n")

# Run encoding with event markers
time.sleep(0.01)  # Small delay to start polling
config = __import__('lmcache.storage_backend.serde.cachegen_basics', fromlist=['CacheGenConfig']).CacheGenConfig.from_model_name("meta-llama/Llama-3.2-1B-Instruct")
key_bins = torch.tensor([32] * num_layers, device='cuda')
value_bins = torch.tensor([32] * num_layers, device='cuda')

encode_with_timeline(kv_cache, config, key_bins, value_bins, num_tokens)

# Stop polling
time.sleep(0.01)  # Capture final state
polling_active = False
poll_thread.join()

print("--- Compression complete ---\n")

# Analyze timeline
with vram_lock:
    # Sort by time
    timeline = sorted(vram_timeline, key=lambda x: x['time'])
    
    # Find peak
    peak_mem = max(t['mem_gb'] for t in timeline)
    peak_time = max(t['mem_gb'] for t in timeline)
    
    # Print timeline
    print(f"{'Time (ms)':<12} {'Type':<8} {'Name':<25} {'VRAM (GB)':<12} {'Delta':<10}")
    print("-"*70)
    
    for t in timeline[:30]:  # First 30 entries
        if t['type'] == 'poll':
            print(f"{t['time']*1000:<12.2f} {'POLL':<8} {'':<25} {t['mem_gb']:<12.4f} {t['mem_delta']:+.4f}")
        else:
            print(f"{t['time']*1000:<12.2f} {'EVENT':<8} {t.get('name',''):<25} {t['mem_gb']:<12.4f} {t['mem_delta']:+.4f}")
    
    if len(timeline) > 30:
        print(f"... ({len(timeline) - 30} more entries)")
    
    # Find VRAM changes
    events = [t for t in timeline if t['type'] == 'event']
    polls = [t for t in timeline if t['type'] == 'poll']
    
    print("\n" + "="*70)
    print("Event Markers Summary")
    print("="*70)
    for e in events:
        print(f"  {e['time']*1000:7.2f}ms | {e['name']:<25} | VRAM: {e['mem_gb']:.4f} GB (+{e['mem_delta']:.4f})")
    
    print("\n" + "="*70)
    print(f"Total timeline entries: {len(timeline)}")
    print(f"  - Polling entries: {len(polls)}")
    print(f"  - Event markers: {len(events)}")
    print(f"Peak VRAM: {peak_mem:.4f} GB")
    print(f"VRAM increase: {peak_mem - base_mem:.4f} GB")
    print("="*70)
