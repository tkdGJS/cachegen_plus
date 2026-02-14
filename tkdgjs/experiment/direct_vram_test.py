#!/usr/bin/env python3
"""Direct nvidia-smi monitoring during request"""
import subprocess
import time
import threading
import os
import sys
import requests

VLLM_BIN = "/home/noslab-gpu/tkdgjs/tkdgjs/bin/vllm"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
PORT = 8000

def wait_for_vllm(timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"http://localhost:{PORT}/v1/models", timeout=5)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False

def monitor_vram(duration=15, interval=0.01):
    """Monitor VRAM using nvidia-smi directly"""
    samples = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=1
        )
        if result.returncode == 0:
            used_mb = float(result.stdout.strip())
            samples.append({
                'time': time.time() - start_time,
                'used_mb': used_mb,
                'used_gb': used_mb / 1024
            })
        time.sleep(interval)
    
    return samples

def send_request():
    """Send a simple request"""
    prompt = "Hello, how are you? " * 500  # ~4000 tokens
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0.0
    }
    
    try:
        r = requests.post(
            f"http://localhost:{PORT}/v1/completions",
            json=payload,
            timeout=30
        )
        return r.status_code == 200
    except Exception as e:
        print(f"Request error: {e}")
        return False

def run_test(mode: str):
    print(f"\n{'='*60}")
    print(f"Testing: {mode}")
    print(f"{'='*60}")
    
    # Start vLLM
    config = "cachegen" if mode == "cachegen" else "torch"
    os.environ["LMCACHE_CONFIG_FILE"] = f"/home/noslab-gpu/tkdgjs/experiment/lmcache_{config}.yaml"
    
    subprocess.run(["pkill", "-9", "-f", "vllm"], stderr=subprocess.DEVNULL)
    time.sleep(3)
    
    cmd = [
        VLLM_BIN, "serve", MODEL,
        "--port", str(PORT),
        "--dtype", "half",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.5",
        "--kv-transfer-config", '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}',
    ]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(30)
    
    if not wait_for_vllm():
        print(f"vLLM not ready for {mode}")
        proc.kill()
        return None
    
    print(f"vLLM ready for {mode}")
    
    # Baseline VRAM
    baseline_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True
    )
    baseline_mb = float(baseline_result.stdout.strip())
    print(f"Baseline VRAM: {baseline_mb:.0f} MB ({baseline_mb/1024:.2f} GB)")
    
    # Start monitoring in background
    samples = []
    def run_monitor():
        nonlocal samples
        samples = monitor_vram(duration=20, interval=0.005)  # 5ms interval
    
    monitor_thread = threading.Thread(target=run_monitor)
    monitor_thread.start()
    
    time.sleep(1)  # Let monitor start
    
    # Send request
    print("Sending request...")
    success = send_request()
    
    # Wait for monitoring to complete
    monitor_thread.join()
    
    proc.kill()
    
    # Analyze results
    if samples:
        used_values = [s['used_gb'] for s in samples]
        min_gb = min(used_values)
        max_gb = max(used_values)
        peak_increase_gb = max_gb - min_gb
        
        print(f"\nResults for {mode}:")
        print(f"  Min VRAM: {min_gb:.4f} GB")
        print(f"  Max VRAM: {max_gb:.4f} GB")
        print(f"  Peak increase: {peak_increase_gb:.4f} GB")
        
        return {
            'mode': mode,
            'baseline_gb': baseline_mb / 1024,
            'min_gb': min_gb,
            'max_gb': max_gb,
            'peak_increase_gb': peak_increase_gb,
            'samples': len(samples)
        }
    
    return None

def main():
    print("="*60)
    print("Direct nvidia-smi VRAM Monitoring")
    print("="*60)
    
    # Test Native
    native_result = run_test("native")
    
    time.sleep(10)
    
    # Test CacheGen
    cachegen_result = run_test("cachegen")
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    if native_result and cachegen_result:
        print(f"\n{'Metric':<25} {'Native':>12} {'CacheGen':>12}")
        print("-"*55)
        print(f"{'Baseline (GB)':<25} {native_result['baseline_gb']:>12.4f} {cachegen_result['baseline_gb']:>12.4f}")
        print(f"{'Min VRAM (GB)':<25} {native_result['min_gb']:>12.4f} {cachegen_result['min_gb']:>12.4f}")
        print(f"{'Max VRAM (GB)':<25} {native_result['max_gb']:>12.4f} {cachegen_result['max_gb']:>12.4f}")
        print(f"{'Peak Increase (GB)':<25} {native_result['peak_increase_gb']:>12.4f} {cachegen_result['peak_increase_gb']:>12.4f}")
        
        diff = cachegen_result['peak_increase_gb'] - native_result['peak_increase_gb']
        print(f"\n{'='*60}")
        print(f"Difference in peak increase: {diff:+.4f} GB")
        
        if diff > 0.1:
            print("RESULT: CacheGen uses MORE VRAM during request")
        elif diff < -0.1:
            print("RESULT: CacheGen uses LESS VRAM during request")
        else:
            print("RESULT: No significant difference")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
