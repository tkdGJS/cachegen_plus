#!/usr/bin/env python3
"""Extended Test - Monitor longer after request"""
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

def run_extended_test(mode: str, prefill_size: int, gpu_mem_util: float):
    print(f"\n{'='*60}")
    print(f"Extended Test: mode={mode}, prefill={prefill_size}")
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
    monitor = FullVRAMMonitorLoop(interval=0.05, port=VLLM_PORT, is_cachegen=is_cachegen)
    monitor.start()
    time.sleep(2)
    
    print(f"[Request] Sending request...")
    success, latency = send_request_and_measure(prefill_size)
    
    print(f"[Monitor] Continuing to monitor for 15 seconds...")
    time.sleep(15)
    
    monitor.stop()
    snapshot = monitor.get_snapshot()
    samples = monitor.get_samples()
    
    snapshot.print_layout(f"VRAM Layout - {mode.upper()}")
    
    vram_file = f"{EXPERIMENT_DIR}/vram_extended_{mode}.jsonl"
    with open(vram_file, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + '\n')
    
    peak_vram = monitor.get_peak_vram()
    
    print(f"\n[Summary]")
    print(f"  Request Success: {success}")
    print(f"  Peak VRAM Increase: {peak_vram:.4f} GB")
    print(f"  Final VRAM: {snapshot.used_vram_gb:.2f} GB")
    print(f"  Samples: {len(samples)}")
    
    stop_vllm()
    
    return {
        "mode": mode,
        "success": success,
        "peak_vram": peak_vram,
        "final_vram": snapshot.used_vram_gb,
        "samples": len(samples)
    }


def main():
    print("="*60)
    print("Extended VRAM Monitoring Test")
    print("="*60)
    
    GPU_MEM_UTIL = 0.5
    PREFILL_SIZE = 4096
    
    # Test Native
    print("\n\n" + "="*60)
    print("TEST: NATIVE MODE")
    print("="*60)
    native = run_extended_test("native", PREFILL_SIZE, GPU_MEM_UTIL)
    
    time.sleep(10)
    
    # Test CacheGen
    print("\n\n" + "="*60)
    print("TEST: CACHEGEN MODE")
    print("="*60)
    cachegen = run_extended_test("cachegen", PREFILL_SIZE, GPU_MEM_UTIL)
    
    # Compare
    print("\n\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    if native and cachegen:
        print(f"\n{'Metric':<25} {'Native':>12} {'CacheGen':>12}")
        print("-"*55)
        print(f"{'Success':<25} {native['success']!s:>12} {cachegen['success']!s:>12}")
        print(f"{'Peak VRAM Increase (GB)':<25} {native['peak_vram']:>12.4f} {cachegen['peak_vram']:>12.4f}")
        print(f"{'Final VRAM (GB)':<25} {native['final_vram']:>12.2f} {cachegen['final_vram']:>12.2f}")
        
        diff = cachegen['peak_vram'] - native['peak_vram']
        print(f"\n{'='*60}")
        print(f"CacheGen Peak VRAM Difference: {diff:+.4f} GB")
        if diff > 0.05:
            print("RESULT: CacheGen shows higher peak VRAM during operation")
        else:
            print("RESULT: No significant difference in peak VRAM")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
