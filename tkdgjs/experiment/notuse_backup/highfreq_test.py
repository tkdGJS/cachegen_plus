#!/usr/bin/env python3
"""High-Frequency VRAM Test - 1ms interval to catch spike"""
import sys
import os
import time
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_sweep_experiment import (
    clear_gpu_processes, clear_lmcache_disk, start_vllm, stop_vllm,
    wait_for_vllm, send_request_and_measure, FullVRAMMonitorLoop,
    VLLM_PORT, MODEL, EXPERIMENT_DIR
)

def run_highfreq_test(mode: str, prefill_size: int, gpu_mem_util: float):
    print(f"\n{'='*60}")
    print(f"High-Freq Test: mode={mode}, prefill={prefill_size}, interval=1ms")
    print(f"{'='*60}")
    
    clear_gpu_processes()
    time.sleep(2)
    clear_lmcache_disk(mode)
    
    vllm_proc = start_vllm(mode, gpu_mem_util)
    if not vllm_proc:
        print(f"[ERROR] Failed to start vLLM")
        return None
    
    if not wait_for_vllm(VLLM_PORT, timeout=180):
        print(f"[ERROR] vLLM not ready")
        stop_vllm()
        return None
    
    is_cachegen = (mode == "cachegen")
    
    # 1ms interval for high-frequency monitoring
    monitor = FullVRAMMonitorLoop(interval=0.001, port=VLLM_PORT, is_cachegen=is_cachegen)
    monitor.start()
    time.sleep(1)
    
    print(f"[Request] Sending request...")
    request_start = time.time()
    success, latency = send_request_and_measure(prefill_size)
    request_end = time.time()
    
    print(f"[Monitor] Continuing to monitor for 10 seconds...")
    time.sleep(10)
    
    monitor.stop()
    snapshot = monitor.get_snapshot()
    samples = monitor.get_samples()
    peak_vram = monitor.get_peak_vram()
    
    snapshot.print_layout(f"VRAM Layout - {mode.upper()}")
    
    vram_file = f"{EXPERIMENT_DIR}/vram_highfreq_{mode}.jsonl"
    with open(vram_file, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + '\n')
    
    # Analyze VRAM changes
    vram_values = [s['used_vram_gb'] for s in samples]
    min_vram = min(vram_values)
    max_vram = max(vram_values)
    vram_range = max_vram - min_vram
    
    print(f"\n[High-Freq Analysis]")
    print(f"  Request duration: {request_end - request_start:.2f}s")
    print(f"  Total samples: {len(samples)}")
    print(f"  Min VRAM: {min_vram:.4f} GB")
    print(f"  Max VRAM: {max_vram:.4f} GB")
    print(f"  Range: {vram_range:.4f} GB")
    print(f"  Peak VRAM increase: {peak_vram:.4f} GB")
    
    # Find samples during request
    request_samples = [s for s in samples if request_start <= s.get('timestamp', 0) <= request_end + 2]
    if request_samples:
        req_vram = [s['used_vram_gb'] for s in request_samples]
        print(f"  During request - Min: {min(req_vram):.4f}, Max: {max(req_vram):.4f}")
    
    stop_vllm()
    
    return {
        "mode": mode,
        "success": success,
        "peak_vram": peak_vram,
        "vram_range": vram_range,
        "min_vram": min_vram,
        "max_vram": max_vram,
        "samples": len(samples)
    }


def main():
    print("="*60)
    print("High-Frequency VRAM Monitoring Test")
    print("Interval: 1ms (0.001s)")
    print("="*60)
    
    GPU_MEM_UTIL = 0.5
    PREFILL_SIZE = 4096
    
    # Test Native
    print("\n\n" + "="*60)
    print("TEST: NATIVE MODE")
    print("="*60)
    native = run_highfreq_test("native", PREFILL_SIZE, GPU_MEM_UTIL)
    
    time.sleep(10)
    
    # Test CacheGen
    print("\n\n" + "="*60)
    print("TEST: CACHEGEN MODE")
    print("="*60)
    cachegen = run_highfreq_test("cachegen", PREFILL_SIZE, GPU_MEM_UTIL)
    
    # Compare
    print("\n\n" + "="*60)
    print("COMPARISON - High Frequency Results")
    print("="*60)
    if native and cachegen:
        print(f"\n{'Metric':<30} {'Native':>12} {'CacheGen':>12}")
        print("-"*60)
        print(f"{'Success':<30} {native['success']!s:>12} {cachegen['success']!s:>12}")
        print(f"{'Min VRAM (GB)':<30} {native['min_vram']:>12.4f} {cachegen['min_vram']:>12.4f}")
        print(f"{'Max VRAM (GB)':<30} {native['max_vram']:>12.4f} {cachegen['max_vram']:>12.4f}")
        print(f"{'VRAM Range (GB)':<30} {native['vram_range']:>12.4f} {cachegen['vram_range']:>12.4f}")
        print(f"{'Peak VRAM Increase (GB)':<30} {native['peak_vram']:>12.4f} {cachegen['peak_vram']:>12.4f}")
        print(f"{'Total Samples':<30} {native['samples']:>12} {cachegen['samples']:>12}")
        
        diff_range = cachegen['vram_range'] - native['vram_range']
        diff_peak = cachegen['peak_vram'] - native['peak_vram']
        
        print(f"\n{'='*60}")
        print(f"CacheGen vs Native:")
        print(f"  VRAM Range Difference: {diff_range:+.4f} GB")
        print(f"  Peak VRAM Difference: {diff_peak:+.4f} GB")
        
        if diff_peak > 0.01:
            print(f"\nRESULT: CacheGen shows HIGHER peak VRAM (+{diff_peak:.4f} GB)")
        elif diff_range > 0.01:
            print(f"\nRESULT: CacheGen shows HIGHER VRAM range (+{diff_range:.4f} GB)")
        else:
            print(f"\nRESULT: No significant difference detected")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
