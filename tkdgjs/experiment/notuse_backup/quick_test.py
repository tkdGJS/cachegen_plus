#!/usr/bin/env python3
"""Quick Validation Test - Native vs CacheGen"""
import sys
import os
import time
import json
import subprocess
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_sweep_experiment import (
    clear_gpu_processes, clear_lmcache_disk, start_vllm, stop_vllm,
    wait_for_vllm, send_request_and_measure, FullVRAMMonitorLoop,
    VLLM_PORT, MODEL, EXPERIMENT_DIR
)

def run_quick_test(mode: str, prefill_size: int, gpu_mem_util: float):
    print(f"\n{'='*60}")
    print(f"Quick Test: mode={mode}, prefill={prefill_size}, gpu_mem={gpu_mem_util}")
    print(f"{'='*60}")
    
    # 1. Clear GPU
    print("\n[1] Clearing GPU...")
    clear_gpu_processes()
    time.sleep(2)
    
    # 2. Clear disk cache
    print("[2] Clearing disk cache...")
    clear_lmcache_disk(mode)
    
    # 3. Start vLLM
    print("[3] Starting vLLM...")
    vllm_proc = start_vllm(mode, gpu_mem_util)
    if not vllm_proc:
        print(f"[ERROR] Failed to start vLLM for {mode}")
        return None
    
    print("[4] Waiting for vLLM to be ready...")
    if not wait_for_vllm(VLLM_PORT, timeout=180):
        print(f"[ERROR] vLLM not ready for {mode}")
        stop_vllm()
        return None
    
    print("[5] vLLM ready, measuring VRAM...")
    
    # 4. Start monitoring
    is_cachegen = (mode == "cachegen")
    monitor = FullVRAMMonitorLoop(interval=0.1, port=VLLM_PORT, is_cachegen=is_cachegen)
    monitor.start()
    time.sleep(2)
    
    # 5. Send request
    print(f"[6] Sending request (prefill={prefill_size})...")
    success, latency = send_request_and_measure(prefill_size)
    
    time.sleep(5)
    
    # 6. Get snapshot
    monitor.stop()
    snapshot = monitor.get_snapshot()
    samples = monitor.get_samples()
    
    # 7. Print layout
    print("\n" + "="*60)
    snapshot.print_layout(f"VRAM Layout - {mode.upper()}")
    
    # 8. Save timeseries
    vram_file = f"{EXPERIMENT_DIR}/vram_quicktest_{mode}_p{prefill_size}.jsonl"
    with open(vram_file, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + '\n')
    print(f"\n[7] Saved timeseries to {vram_file}")
    
    # 9. Stop vLLM
    stop_vllm()
    
    # 10. Print summary
    print(f"\n{'='*60}")
    print(f"Result Summary - {mode.upper()}")
    print(f"{'='*60}")
    print(f"  Request Success: {success}")
    if latency:
        print(f"  TTFT: {latency.get('ttft_sec', 'N/A')}s")
        print(f"  TTLT: {latency.get('ttlt_sec', 'N/A')}s")
    print(f"  Used VRAM: {snapshot.used_vram_gb:.2f} GB")
    print(f"  KV Allocated: {snapshot.vllm_kv_cache_allocated_gb:.2f} GB")
    print(f"  KV Used: {snapshot.vllm_kv_cache_used_gb:.2f} GB")
    print(f"  Sum Validated: {snapshot.sum_validated_gb:.2f} GB")
    print(f"  Sum Diff: {snapshot.sum_diff_gb:+.2f} GB")
    if is_cachegen:
        print(f"  CacheGen Total: {snapshot.cachegen_total_gb:.4f} GB")
    print(f"{'='*60}")
    
    return {
        "mode": mode,
        "success": success,
        "latency": latency,
        "snapshot": snapshot
    }


def main():
    print("="*60)
    print("Quick Validation Test")
    print("GPU Util: 0.5, Prefill: 4096")
    print("Modes: native, cachegen")
    print("="*60)
    
    GPU_MEM_UTIL = 0.5
    PREFILL_SIZE = 4096
    
    # Test Native
    print("\n\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█  TEST 1: NATIVE MODE                                  █")
    print("█" + " "*58 + "█")
    print("█"*60)
    
    native_result = run_quick_test("native", PREFILL_SIZE, GPU_MEM_UTIL)
    
    time.sleep(10)
    
    # Test CacheGen
    print("\n\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█  TEST 2: CACHEGEN MODE                                 █")
    print("█" + " "*58 + "█")
    print("█"*60)
    
    cachegen_result = run_quick_test("cachegen", PREFILL_SIZE, GPU_MEM_UTIL)
    
    # Compare results
    print("\n\n" + "="*60)
    print("COMPARISON: Native vs CacheGen")
    print("="*60)
    
    if native_result and cachegen_result:
        nat = native_result["snapshot"]
        cge = cachegen_result["snapshot"]
        
        print(f"\n{'Metric':<25} {'Native':>12} {'CacheGen':>12} {'Diff':>12}")
        print("-"*65)
        print(f"{'Used VRAM (GB)':<25} {nat.used_vram_gb:>12.2f} {cge.used_vram_gb:>12.2f} {cge.used_vram_gb - nat.used_vram_gb:>+12.2f}")
        print(f"{'KV Allocated (GB)':<25} {nat.vllm_kv_cache_allocated_gb:>12.2f} {cge.vllm_kv_cache_allocated_gb:>12.2f} {cge.vllm_kv_cache_allocated_gb - nat.vllm_kv_cache_allocated_gb:>+12.2f}")
        print(f"{'KV Used (GB)':<25} {nat.vllm_kv_cache_used_gb:>12.2f} {cge.vllm_kv_cache_used_gb:>12.2f} {cge.vllm_kv_cache_used_gb - nat.vllm_kv_cache_used_gb:>+12.2f}")
        print(f"{'Model Weights (GB)':<25} {nat.model_weights_gb:>12.2f} {cge.model_weights_gb:>12.2f} {cge.model_weights_gb - nat.model_weights_gb:>+12.2f}")
        print(f"{'Activation (GB)':<25} {nat.activation_tensors_gb:>12.2f} {cge.activation_tensors_gb:>12.2f} {cge.activation_tensors_gb - nat.activation_tensors_gb:>+12.2f}")
        print(f"{'CUDA Runtime (GB)':<25} {nat.cuda_runtime_gb:>12.2f} {cge.cuda_runtime_gb:>12.2f} {cge.cuda_runtime_gb - nat.cuda_runtime_gb:>+12.2f}")
        print(f"{'CacheGen Total (GB)':<25} {'N/A':>12} {cge.cachegen_total_gb:>12.4f} {'N/A':>12}")
        print(f"{'Sum Diff (GB)':<25} {nat.sum_diff_gb:>+12.2f} {cge.sum_diff_gb:>+12.2f} {'N/A':>12}")
        
        vram_diff = cge.used_vram_gb - nat.used_vram_gb
        print(f"\n{'='*60}")
        if vram_diff > 0.1:
            print(f"RESULT: CacheGen uses MORE VRAM than Native (+{vram_diff:.2f} GB)")
            print("HYPOTHESIS CONFIRMED: Compression buffers increase VRAM usage")
        elif vram_diff < -0.1:
            print(f"RESULT: CacheGen uses LESS VRAM than Native ({vram_diff:.2f} GB)")
            print("HYPOTHESIS REJECTED: Compression reduces overall VRAM")
        else:
            print(f"RESULT: No significant VRAM difference ({vram_diff:.2f} GB)")
            print("INCONCLUSIVE: Need more detailed measurement")
        print(f"{'='*60}")
    
    print("\nTest files saved:")
    print(f"  - {EXPERIMENT_DIR}/vram_quicktest_native_p4096.jsonl")
    print(f"  - {EXPERIMENT_DIR}/vram_quicktest_cachegen_p4096.jsonl")


if __name__ == "__main__":
    main()
