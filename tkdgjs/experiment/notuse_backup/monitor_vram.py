#!/usr/bin/env python3
"""
VRAM Time Series Monitor for LMCache Compression Testing
Collects VRAM usage over time during vLLM inference with LMCache
"""

import json
import time
import subprocess
import os
import sys
from datetime import datetime

def get_vram_usage():
    """Get current VRAM usage in GB"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            mem_mb = float(result.stdout.strip().split('\n')[0])
            return mem_mb / 1024.0  # Convert to GB
    except Exception as e:
        print(f"Error getting VRAM: {e}")
    return None

def get_lmcache_vram_log():
    """Read LMCache VRAM log"""
    log_file = os.environ.get('LMCACHE_VRAM_LOG_FILE', '/tmp/lmcache_vram.log')
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                return f.read()
    except:
        pass
    return ""

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "cachegen"
    output_file = f"/tmp/vram_timeseries_{mode}.jsonl"
    
    print(f"Starting VRAM monitoring for {mode} mode...")
    print(f"Output file: {output_file}")
    
    results = []
    start_time = time.time()
    
    # Baseline VRAM
    baseline = get_vram_usage()
    if baseline:
        results.append({
            "time": 0,
            "vram_gb": baseline,
            "event": "baseline",
            "mode": mode
        })
        print(f"Baseline VRAM: {baseline:.4f} GB")
    
    # Send request and monitor
    print("\nSending test request...")
    
    # Record VRAM before request
    vram_before = get_vram_usage()
    
    # Send request via curl
    cmd = """curl -s http://localhost:8005/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "prompt": "Write a detailed story about a dragon who loves to cook.",
        "max_tokens": 200,
        "temperature": 0.7
      }'"""
    
    # Monitor during request
    import threading
    import queue
    
    monitoring = True
    vram_data = queue.Queue()
    
    def monitor_vram():
        while monitoring:
            vram = get_vram_usage()
            if vram:
                vram_data.put({
                    "time": time.time() - start_time,
                    "vram_gb": vram
                })
            time.sleep(0.1)
    
    monitor_thread = threading.Thread(target=monitor_vram)
    monitor_thread.start()
    
    # Execute request
    os.system(cmd + " > /dev/null 2>&1")
    
    # Wait a bit more
    time.sleep(2)
    monitoring = False
    monitor_thread.join()
    
    # Collect VRAM data
    while not vram_data.empty():
        results.append(vram_data.get())
    
    # Final VRAM
    vram_after = get_vram_usage()
    if vram_after:
        results.append({
            "time": time.time() - start_time,
            "vram_gb": vram_after,
            "event": "final",
            "mode": mode
        })
    
    # Get LMCache VRAM log
    lmcache_log = get_lmcache_vram_log()
    
    # Save results
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    print(f"\n=== Results ===")
    print(f"Baseline: {baseline:.4f} GB" if baseline else "N/A")
    print(f"Peak: {max(r['vram_gb'] for r in results):.4f} GB" if results else "N/A")
    print(f"Final: {vram_after:.4f} GB" if vram_after else "N/A")
    
    # Print LMCache compression info
    if lmcache_log:
        print(f"\n=== LMCache VRAM Log ===")
        print(lmcache_log[-1000:])  # Last 1000 chars
    
    print(f"\nData saved to: {output_file}")

if __name__ == "__main__":
    main()
