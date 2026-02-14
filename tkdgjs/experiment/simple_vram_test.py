#!/usr/bin/env python3
import subprocess
import time
import threading

def monitor_vram():
    results = []
    for _ in range(200):
        r = subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i','0'], capture_output=True, text=True)
        if r.returncode == 0:
            results.append(float(r.stdout.strip()))
        time.sleep(0.01)
    if results:
        return min(results), max(results), max(results)-min(results)
    return 0, 0, 0

# Start vLLM with lower utilization
cmd = [
    '/home/noslab-gpu/tkdgjs/tkdgjs/bin/vllm', 'serve', 'meta-llama/Llama-3.2-1B-Instruct',
    '--port', '8000', '--dtype', 'half', '--max-model-len', '8192',
    '--gpu-memory-utilization', '0.3',
    '--kv-transfer-config', '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
]

print("Starting vLLM with gpu_memory_utilization=0.3...")
proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Waiting 45s for vLLM...")
time.sleep(45)

# Check if ready
r = subprocess.run(['curl', '-s', 'http://localhost:8000/v1/models'], capture_output=True)
if r.returncode != 0:
    print("vLLM not ready!")
    proc.kill()
    exit(1)

print("vLLM ready! Testing VRAM during requests...")

for i in range(3):
    print(f"\n=== Round {i+1} ===")
    min_v, max_v, range_v = monitor_vram()
    print(f"Before request: {min_v:.0f} MB")
    
    # Send request
    r = subprocess.run(['curl', '-s', '-X', 'POST', 'http://localhost:8000/v1/completions',
                   '-H', 'Content-Type: application/json',
                   '-d', '{"model":"meta-llama/Llama-3.2-1B-Instruct","prompt":"Hello world","max_tokens":5}'],
                  capture_output=True, timeout=30)
    
    min_v, max_v, range_v = monitor_vram()
    print(f"During request: Min={min_v:.0f} MB, Max={max_v:.0f} MB, Range={range_v:.0f} MB")
    time.sleep(2)

proc.kill()
print("\nDone!")
