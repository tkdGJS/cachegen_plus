#!/usr/bin/env python3
"""Test 4 different methods to measure temporary VRAM during compression"""

import subprocess
import time
import sys
import os
import threading

print("="*60)
print("Testing 4 Methods for VRAM Spike Detection")
print("="*60)

# Method 1: CUDA Memory Events (callback approach)
print("\n[Method 1] CUDA Memory Events API")
print("-"*40)

try:
    result = subprocess.run([
        '/home/noslab-gpu/tkdgjs/tkdgjs/bin/python', '-c',
        '''
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Check if allocator callbacks exist
if hasattr(torch.cuda.memory, "_set_allocator_callbacks"):
    print("OK: _set_allocator_callbacks exists")
else:
    print("NOT FOUND: _set_allocator_callbacks")
    
# Check for memory snapshot
if hasattr(torch.cuda.memory, "memory_snapshot"):
    print("OK: memory_snapshot exists")
else:
    print("NOT FOUND: memory_snapshot")
'''
    ], capture_output=True, text=True, timeout=30)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
except Exception as e:
    print(f"Error: {e}")

# Method 2: torch.cuda.memory_allocated()
print("\n[Method 2] torch.cuda.memory_allocated()")
print("-"*40)

try:
    result = subprocess.run([
        '/home/noslab-gpu/tkdgjs/tkdgjs/bin/python', '-c',
        '''
import torch
print(f"memory_allocated exists: {hasattr(torch.cuda, 'memory_allocated')}")
print(f"max_memory_allocated exists: {hasattr(torch.cuda, 'max_memory_allocated')}")
print(f"reset_peak_memory_stats exists: {hasattr(torch.cuda, 'reset_peak_memory_stats')}")

# Try simple allocation test
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    x = torch.zeros(10000000, device="cuda")  # ~40MB
    allocated = torch.cuda.memory_allocated() / 1024**2
    print(f"Allocated 40MB tensor, memory_allocated: {allocated:.2f} MB")
    del x
    torch.cuda.synchronize()
    print("OK: memory_allocated works!")
else:
    print("CUDA not available")
'''
    ], capture_output=True, text=True, timeout=30)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
except Exception as e:
    print(f"Error: {e}")

# Method 3: NVTX
print("\n[Method 3] NVTX Ranges")
print("-"*40)

try:
    result = subprocess.run([
        '/home/noslab-gpu/tkdgjs/tkdgjs/bin/python', '-c',
        '''
import torch
try:
    import nvtx
    print(f"NVTX version: {nvtx.__version__}")
    print("OK: nvtx available")
    
    # Test nvtx.range
    with nvtx.range("test"):
        x = torch.zeros(1000000, device="cuda")
        del x
    print("OK: nvtx.range works!")
except ImportError:
    print("NOT FOUND: nvtx")
except Exception as e:
    print(f"Error: {e}")
'''
    ], capture_output=True, text=True, timeout=30)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
except Exception as e:
    print(f"Error: {e}")

# Method 4: Kernel-level tracing
print("\n[Method 4] Kernel-level tracing")
print("-"*40)

# Check if we can use nvidia-smi with high frequency
try:
    result = subprocess.run([
        'nvidia-smi', '--query-gpu=timestamp,memory.used',
        '--format=csv,noheader', '-i', '0', '-l', '1'
    ], capture_output=True, text=True, timeout=3)
    print("OK: nvidia-smi with 1ms interval works")
    print(f"Sample output: {result.stdout.split(chr(10))[0]}")
except Exception as e:
    print(f"Error: {e}")

# Check for perfetto/tracing tools
try:
    result = subprocess.run(['which', 'perfetto'], capture_output=True)
    if result.returncode == 0:
        print("FOUND: perfetto")
    else:
        print("NOT FOUND: perfetto")
except:
    pass

print("\n" + "="*60)
print("Summary of Available Methods")
print("="*60)
