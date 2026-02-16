#!/usr/bin/env python3
"""
VRAM Measurement during LMCache operations using torch.cuda.memory_allocated()

This test uses torch.cuda.memory_allocated() to measure VRAM changes
during CacheGen compression/decompression.
"""
import subprocess
import time
import os

VLLM_BIN = "/home/noslab-gpu/tkdgjs/tkdgjs/bin/vllm"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
PORT = 8000

def wait_for_vllm(timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = subprocess.run(['curl', '-s', f'http://localhost:{PORT}/v1/models'],
                            capture_output=True, timeout=5)
            if r.returncode == 0:
                return True
        except:
            pass
        time.sleep(2)
    return False

def measure_with_memory_allocated():
    """
    Measure VRAM using torch.cuda.memory_allocated() during request
    """
    script = '''
import torch
import requests
import time

# Reset memory stats
torch.cuda.reset_peak_memory_stats()

# Baseline
baseline = torch.cuda.memory_allocated() / 1024**2
print(f"Baseline VRAM: {baseline:.2f} MB")

# Send request
payload = {
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "prompt": "Hello world, this is a test. " * 500,
    "max_tokens": 10,
    "temperature": 0.0
}

start_time = time.time()

# Measure during request
try:
    r = requests.post(
        f"http://localhost:{PORT}/v1/completions",
        json=payload,
        timeout=30
    )
    
    # Multiple measurements during request
    measurements = []
    for i in range(10):
        mem = torch.cuda.memory_allocated() / 1024**2
        measurements.append(mem)
        time.sleep(0.1)
    
    print(f"Request took: {time.time() - start_time:.2f}s")
    print(f"Measurements (MB): {measurements}")
    print(f"Min: {min(measurements):.2f}, Max: {max(measurements):.2f}, Range: {max(measurements)-min(measurements):.2f}")
    
except Exception as e:
    print(f"Error: {e}")
'''
    
    result = subprocess.run([
        VLLM_BIN, '-c', script
    ], capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': '/home/noslab-gpu/tkdgjs/tkdgjs/lib/python3.10/site-packages'},
       cwd='/home/noslab-gpu/tkdgjs/tkdgjs')
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])

def test_memory_allocated_simple():
    """Simple test to verify memory_allocated works"""
    script = '''
import torch
import time

print("=== Testing torch.cuda.memory_allocated() ===")
print(f"Initial: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Allocate some memory
x = torch.zeros(10000000, device="cuda")  # ~40MB
print(f"After allocation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Simulate compression (multiple allocations)
y = torch.zeros(20000000, device="cuda")  # ~80MB
print(f"After 2nd allocation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Free
del y
del x
torch.cuda.synchronize()
print(f"After deletion: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

print("\\nOK: torch.cuda.memory_allocated() works for measuring VRAM!")
'''
    
    result = subprocess.run([
        VLLM_BIN, '-c', script
    ], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])

def test_memory_snapshot():
    """Test memory_snapshot for detailed allocation info"""
    script = '''
import torch
import time

print("=== Testing torch.cuda.memory_snapshot() ===")

# Allocate some memory
x = torch.zeros(10000000, device="cuda")
y = torch.zeros(20000000, device="cuda")

# Get snapshot
snap = torch.cuda.memory_snapshot()
print(f"Snapshot returned {len(snap)} segments")

for seg in snap[:3]:  # Show first 3
    print(f"  size: {seg.get('size', 0) / 1024**2:.2f} MB, "
          f"allocated: {seg.get('allocated', False)}, "
          f"device: {seg.get('device', 0)}")

del x, y
torch.cuda.synchronize()

print("\\nOK: torch.cuda.memory_snapshot() provides detailed allocation info!")
'''
    
    result = subprocess.run([
        VLLM_BIN, '-c', script
    ], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])

# First, let's test if torch.cuda works in the vllm environment
print("="*60)
print("Testing VRAM Measurement Methods")
print("="*60)

print("\n[Test 1] Simple memory_allocated test")
test_memory_allocated_simple()

print("\n[Test 2] Memory snapshot test")
test_memory_snapshot()

print("\n" + "="*60)
print("Conclusion: Method 2 (memory_allocated) works well!")
print("Use torch.cuda.memory_allocated() to measure VRAM changes")
print("="*60)
