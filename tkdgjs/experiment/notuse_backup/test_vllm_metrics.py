#!/usr/bin/env python3
"""
Quick test: Compare Native vs CacheGen VRAM during 4096 token request
Using vLLM internal metrics
"""
import requests
import time
import subprocess
import json
import os
import signal
import sys
from openai import OpenAI

VLLM_PORT = 8000
MODEL = "meta-llama/Llama-3.2-1B-Instruct"

def start_vllm(mode, gpu_mem_util=0.7):
    config = f"/home/noslab-gpu/tkdgjs/experiment/lmcache_{mode}.yaml"
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(VLLM_PORT),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--config", config,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(30)
    return proc

def stop_vllm(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except:
        proc.kill()

def get_vllm_metrics():
    try:
        resp = requests.get(f"http://localhost:{VLLM_PORT}/metrics", timeout=2)
        if resp.status_code != 200:
            return {}
        
        text = resp.text
        metrics = {}
        
        for line in text.splitlines():
            if not line or line.startswith('#'):
                continue
            parts = line.rsplit(' ', 1)
            if len(parts) == 2:
                metrics[parts[0]] = float(parts[1])
        
        return metrics
    except:
        return {}

def get_custom_metrics():
    metrics = get_vllm_metrics()
    
    kv_usage_perc = metrics.get('vllm:kv_cache_usage_perc', 0)
    num_running = metrics.get('vllm:num_running_requests', 0)
    num_waiting = metrics.get('vllm:num_waiting_requests', 0)
    num_swapped = metrics.get('vllm:num_swapped_requests', 0)
    
    num_blocks_total = int(metrics.get('vllm:num_gpu_blocks', 0))
    num_blocks_free = int(metrics.get('vllm:num_gpu_blocks_free', 0))
    num_blocks_used = num_blocks_total - num_blocks_free
    
    return {
        "kv_usage_perc": kv_usage_perc,
        "num_running": num_running,
        "num_waiting": num_waiting,
        "num_swapped": num_swapped,
        "num_blocks_used": num_blocks_used,
        "num_blocks_total": num_blocks_total,
    }

def send_request(prompt, max_tokens=512):
    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{VLLM_PORT}/v1")
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
    )
    return response

def test_mode(mode, prefill_tokens=4096):
    print(f"\n{'='*60}")
    print(f"Testing {mode.upper()} mode with {prefill_tokens} tokens")
    print(f"{'='*60}")
    
    print("[1] Starting vLLM...")
    proc = start_vllm(mode)
    print("[2] vLLM started, waiting for metrics...")
    
    time.sleep(5)
    
    baseline = get_custom_metrics()
    print(f"    Baseline: KV_usage={baseline['kv_usage_perc']:.4f}%, running={baseline['num_running']}, blocks_used={baseline['num_blocks_used']}")
    
    prompt = "Hello, how are you? " * (prefill_tokens // 4)
    print(f"[3] Sending request ({prefill_tokens} tokens)...")
    
    measurements = []
    
    def poll_metrics():
        for _ in range(300):
            m = get_custom_metrics()
            measurements.append(m)
            if m['num_running'] > 0:
                print(f"    During request: KV_usage={m['kv_usage_perc']:.4f}%, running={m['num_running']}, blocks_used={m['num_blocks_used']}")
            time.sleep(0.01)
    
    import threading
    poll_thread = threading.Thread(target=poll_metrics)
    poll_thread.start()
    
    try:
        response = send_request(prompt)
        print(f"[4] Response received, latency={response.usage.total_tokens} tokens")
    except Exception as e:
        print(f"[4] Error: {e}")
    
    time.sleep(2)
    
    poll_thread.join()
    
    if measurements:
        max_kv = max(m['kv_usage_perc'] for m in measurements)
        max_running = max(m['num_running'] for m in measurements)
        max_blocks = max(m['num_blocks_used'] for m in measurements)
        print(f"[5] Max during request: KV_usage={max_kv:.4f}%, running={max_running}, blocks_used={max_blocks}")
    
    print("[6] Stopping vLLM...")
    stop_vllm(proc)
    
    return {
        "mode": mode,
        "baseline_kv_usage": baseline['kv_usage_perc'],
        "max_kv_usage": max(measurements, key=lambda m: m['kv_usage_perc'])['kv_usage_perc'] if measurements else 0,
        "baseline_blocks": baseline['num_blocks_used'],
        "max_blocks": max(m['num_blocks_used'] for m in measurements) if measurements else 0,
    }

def main():
    from openai import OpenAI
    
    print("VRAM Metrics Comparison Test")
    print("="*60)
    
    results = []
    
    for mode in ["native", "cachegen"]:
        result = test_mode(mode, prefill_tokens=4096)
        results.append(result)
        time.sleep(10)
    
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    for r in results:
        print(f"\n{r['mode'].upper()}:")
        print(f"  Baseline KV_usage: {r['baseline_kv_usage']:.4f}%")
        print(f"  Max KV_usage: {r['max_kv_usage']:.4f}%")
        print(f"  Baseline blocks: {r['baseline_blocks']}")
        print(f"  Max blocks: {r['max_blocks']}")
    
    native = results[0]
    cachegen = results[1]
    
    print("\n" + "="*60)
    print("DIFFERENCE")
    print("="*60)
    print(f"KV_usage diff: {cachegen['max_kv_usage'] - native['max_kv_usage']:+.4f}%")
    print(f"Blocks diff: {cachegen['max_blocks'] - native['max_blocks']:+.0f}")

if __name__ == "__main__":
    main()
