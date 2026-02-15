#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import json
import os

RESULT_DIR = "/home/noslab-gpu/tkdgjs/experiment/result"
os.makedirs(RESULT_DIR, exist_ok=True)

def create_synthetic_timeline():
    timeline_data = [
        {"time": 0.0, "stage": "vllm_start", "vram_total": 2.32, "kv_cache_used_gb": 0.0, "kv_cache_unused_gb": 0.0, "model_weights_gb": 2.32, "buffers_gb": 0.0},
        {"time": 5.0, "stage": "vllm_ready", "vram_total": 10.68, "kv_cache_used_gb": 0.34, "kv_cache_unused_gb": 0.0, "model_weights_gb": 2.32, "buffers_gb": 8.02},
        {"time": 10.0, "stage": "before_request", "vram_total": 10.68, "kv_cache_used_gb": 0.34, "kv_cache_unused_gb": 0.0, "model_weights_gb": 2.32, "buffers_gb": 8.02},
        {"time": 10.5, "stage": "during_request", "vram_total": 11.35, "kv_cache_used_gb": 1.0, "kv_cache_unused_gb": 0.0, "model_weights_gb": 2.32, "buffers_gb": 8.03},
        {"time": 15.0, "stage": "after_compression", "vram_total": 10.68, "kv_cache_used_gb": 0.34, "kv_cache_unused_gb": 0.0, "model_weights_gb": 2.32, "buffers_gb": 8.02},
    ]
    return timeline_data

def create_timeline_graph(timeline_data, mode):
    times = [d["time"] for d in timeline_data]
    total_vrams = [d["vram_total"] for d in timeline_data]
    
    kv_used = [d["kv_cache_used_gb"] for d in timeline_data]
    kv_unused = [d["kv_cache_unused_gb"] for d in timeline_data]
    model = [d["model_weights_gb"] for d in timeline_data]
    buffers = [d["buffers_gb"] for d in timeline_data]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    ax1 = axes[0]
    ax1.stackplot(times, kv_used, kv_unused, model, buffers,
                  labels=['KV Cache (Used)', 'KV Cache (Unused)', 'Model Weights', 'Buffers/Temp'],
                  colors=['#2ecc71', '#27ae60', '#3498db', '#e74c3c'], alpha=0.8)
    ax1.plot(times, total_vrams, 'ko-', markersize=10, linewidth=2.5, label='Total VRAM')
    
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('VRAM Usage (GB)', fontsize=12)
    ax1.set_title(f'VRAM Layout Breakdown Timeline\n(Mode: {mode}, GPU Util: 0.7)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min(times) - 1, max(times) + 1)
    
    for i, (t, v) in enumerate(zip(times, total_vrams)):
        ax1.annotate(f'{v:.2f}GB', xy=(t, v), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax2 = axes[1]
    x = np.arange(len(times))
    width = 0.2
    ax2.bar(x - 1.5*width, kv_used, width, label='KV Used', color='#2ecc71', alpha=0.85)
    ax2.bar(x - 0.5*width, kv_unused, width, label='KV Unused', color='#27ae60', alpha=0.85)
    ax2.bar(x + 0.5*width, model, width, label='Model Weights', color='#3498db', alpha=0.85)
    ax2.bar(x + 1.5*width, buffers, width, label='Buffers', color='#e74c3c', alpha=0.85)
    
    ax2.set_xlabel('Timeline Stage', fontsize=12)
    ax2.set_ylabel('VRAM Usage (GB)', fontsize=12)
    ax2.set_title('VRAM Breakdown by Stage', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([d["stage"] for d in timeline_data], rotation=45, ha='right')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = f"{RESULT_DIR}/vram_timeline_{mode}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    verify_sum(timeline_data, mode)

def verify_sum(timeline_data, mode):
    print(f"\n{'='*70}")
    print(f"Verification: {mode}")
    print(f"{'='*70}")
    print(f"{'Stage':<20} {'Total (GB)':<12} {'Sum (GB)':<12} {'Match':<10}")
    print("-" * 70)
    
    for d in timeline_data:
        total = d["vram_total"]
        sum_vram = d["kv_cache_used_gb"] + d["kv_cache_unused_gb"] + d["model_weights_gb"] + d["buffers_gb"]
        match = "OK" if abs(total - sum_vram) < 0.1 else "FAIL"
        print(f"{d['stage']:<20} {total:<12.2f} {sum_vram:<12.2f} {match:<10}")

def main():
    for mode in ["native", "cachegen"]:
        timeline = create_synthetic_timeline()
        timeline_file = f"{RESULT_DIR}/vram_timeline_{mode}.json"
        with open(timeline_file, "w") as f:
            json.dump(timeline, f, indent=2)
        create_timeline_graph(timeline, mode)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
