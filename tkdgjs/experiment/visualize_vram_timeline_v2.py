#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import json
import os

RESULT_DIR = "/home/noslab-gpu/tkdgjs/experiment/result"
os.makedirs(RESULT_DIR, exist_ok=True)

def create_comparison_graph():
    native = [
        {"time": 0.0, "stage": "cleanup", "vram_total": 0.0},
        {"time": 2.0, "stage": "start_vllm", "vram_total": 2.32},
        {"time": 15.0, "stage": "vllm_ready", "vram_total": 10.68},
        {"time": 18.0, "stage": "send_request", "vram_total": 10.68},
        {"time": 18.5, "stage": "during", "vram_total": 11.68},
        {"time": 25.0, "stage": "after", "vram_total": 10.68},
    ]
    
    cachegen = [
        {"time": 0.0, "stage": "cleanup", "vram_total": 0.0},
        {"time": 2.0, "stage": "start_vllm", "vram_total": 2.32},
        {"time": 15.0, "stage": "vllm_ready", "vram_total": 10.68},
        {"time": 18.0, "stage": "send_request", "vram_total": 10.68},
        {"time": 18.5, "stage": "during", "vram_total": 12.35},
        {"time": 25.0, "stage": "after", "vram_total": 10.68},
    ]
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    native_times = [d["time"] for d in native]
    native_vrams = [d["vram_total"] for d in native]
    
    cachegen_times = [d["time"] for d in cachegen]
    cachegen_vrams = [d["vram_total"] for d in cachegen]
    
    ax1 = axes[0]
    ax1.plot(native_times, native_vrams, 'b-o', markersize=12, linewidth=2.5, label='Native (KV Copy Only)')
    ax1.plot(cachegen_times, cachegen_vrams, 'r-o', markersize=12, linewidth=2.5, label='CacheGen (Compression)')
    
    for t, v in zip(native_times, native_vrams):
        ax1.annotate(f'{v:.1f}GB', xy=(t, v), xytext=(8, 8), textcoords='offset points', fontsize=10, fontweight='bold', color='blue')
    for t, v in zip(cachegen_times, cachegen_vrams):
        ax1.annotate(f'{v:.1f}GB', xy=(t, v), xytext=(8, 8), textcoords='offset points', fontsize=10, fontweight='bold', color='red')
    
    ax1.axvspan(15, 25, alpha=0.1, color='yellow', label='Experiment Active Period')
    ax1.axvspan(18, 19, alpha=0.3, color='green', label='Request Processing')
    
    ax1.set_xlabel('Time (seconds)', fontsize=14)
    ax1.set_ylabel('VRAM Usage (GB)', fontsize=14)
    ax1.set_title('VRAM Timeline: Native vs CacheGen\n(Full Experiment Timeline with Sleep)', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 30)
    
    ax2 = axes[1]
    
    stages = ["cleanup", "start_vllm", "vllm_ready", "send_request", "during", "after"]
    x = np.arange(len(stages))
    width = 0.35
    
    native_vals = [0.0, 2.32, 10.68, 10.68, 11.68, 10.68]
    cachegen_vals = [0.0, 2.32, 10.68, 10.68, 12.35, 10.68]
    
    bars1 = ax2.bar(x - width/2, native_vals, width, label='Native', color='#3498db', alpha=0.85)
    bars2 = ax2.bar(x + width/2, cachegen_vals, width, label='CacheGen', color='#e74c3c', alpha=0.85)
    
    for i, (n, c) in enumerate(zip(native_vals, cachegen_vals)):
        diff = c - n
        color = 'green' if diff > 0 else 'blue'
        ax2.annotate(f'{diff:+.1f}GB', xy=(i, max(n, c) + 0.2), ha='center', fontsize=11, fontweight='bold', color=color)
    
    ax2.set_xlabel('Stage', fontsize=14)
    ax2.set_ylabel('VRAM Usage (GB)', fontsize=14)
    ax2.set_title('VRAM Comparison by Stage', fontsize=16, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages, fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = f"{RESULT_DIR}/vram_timeline_comparison_v2.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    create_comparison_graph()
    print("Done!")
