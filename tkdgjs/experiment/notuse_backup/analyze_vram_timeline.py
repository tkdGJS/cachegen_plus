#!/usr/bin/env python3
"""
VRAM Timeline Analysis - Stage별 비교 그래프
x축: Stage (before_start → after_start → before_request → during_request → after_compression)
y축: VRAM (GB)
"""
import json
import glob
import os

EXPERIMENT_DIR = "/home/noslab-gpu/tkdgjs/experiment/backup_20260214"

def load_all_results():
    results = []
    for f in glob.glob(f"{EXPERIMENT_DIR}/timeline_*.jsonl"):
        with open(f, "r") as fp:
            for line in fp:
                data = json.loads(line)
                if data.get("type") == "result":
                    results.append(data)
    return results

def analyze_by_gpu_util(results):
    stages = ["before_start", "after_start", "before_request", "during_request", "after_compression"]
    
    print("\n" + "="*80)
    print("VRAM Analysis by GPU Utilization")
    print("="*80)
    
    for gpu_util in [0.3, 0.5, 0.7, 0.9]:
        print(f"\n### GPU Util: {gpu_util}")
        print(f"{'Prefill':<10} {'Mode':<10} {'before_start':<12} {'after_start':<14} {'before_req':<12} {'during_req':<12} {'after_comp':<12}")
        print("-" * 90)
        
        for prefill in [128, 256, 512, 1024, 2048]:
            for mode in ["native", "cachegen"]:
                key = f"{mode}_p{prefill}_gm{gpu_util}.jsonl"
                for r in results:
                    if r["mode"] == mode and r["prefill_size"] == prefill and r["gpu_util"] == gpu_util:
                        print(f"{prefill:<10} {mode:<10} "
                              f"{r.get('vram_before_start_gb', 0):<12.2f} "
                              f"{r.get('vram_after_start_gb', 0):<14.2f} "
                              f"{r.get('vram_before_request_gb', 0):<12.2f} "
                              f"{r.get('vram_during_request_gb', 0):<12.2f} "
                              f"{r.get('vram_after_compression_gb', 0):<12.2f}")

def create_csv_for_graph():
    results = load_all_results()
    
    csv_file = f"{EXPERIMENT_DIR}/vram_stage_comparison.csv"
    with open(csv_file, "w") as f:
        f.write("gpu_util,prefill,mode,stage,vram_gb\n")
        
        for r in results:
            stages = [
                ("before_start", r.get("vram_before_start_gb", 0)),
                ("after_start", r.get("vram_after_start_gb", 0)),
                ("before_request", r.get("vram_before_request_gb", 0)),
                ("during_request", r.get("vram_during_request_gb", 0)),
                ("after_compression", r.get("vram_after_compression_gb", 0)),
            ]
            for stage, vram in stages:
                f.write(f"{r['gpu_util']},{r['prefill_size']},{r['mode']},{stage},{vram}\n")
    
    print(f"\nCSV saved to: {csv_file}")
    return csv_file

def compare_native_vs_cachegen(results):
    print("\n" + "="*80)
    print("Native vs CacheGen VRAM Comparison (After Start)")
    print("="*80)
    
    print(f"\n{'GPU Util':<12} {'Prefill':<10} {'Native (GB)':<14} {'CacheGen (GB)':<14} {'Difference':<12}")
    print("-" * 70)
    
    for gpu_util in [0.3, 0.5, 0.7, 0.9]:
        for prefill in [128, 256, 512, 1024, 2048]:
            native_vram = None
            cachegen_vram = None
            
            for r in results:
                if r["prefill_size"] == prefill and r["gpu_util"] == gpu_util:
                    if r["mode"] == "native":
                        native_vram = r.get("vram_after_start_gb", 0)
                    elif r["mode"] == "cachegen":
                        cachegen_vram = r.get("vram_after_start_gb", 0)
            
            if native_vram is not None and cachegen_vram is not None:
                diff = cachegen_vram - native_vram
                print(f"{gpu_util:<12} {prefill:<10} {native_vram:<14.2f} {cachegen_vram:<14.2f} {diff:+.2f}")

if __name__ == "__main__":
    results = load_all_results()
    print(f"Loaded {len(results)} results")
    
    analyze_by_gpu_util(results)
    compare_native_vs_cachegen(results)
    create_csv_for_graph()
