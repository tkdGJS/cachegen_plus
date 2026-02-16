#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os

RESULT_DIR = "/home/noslab-gpu/tkdgjs/experiment/result"
os.makedirs(RESULT_DIR, exist_ok=True)

native_data = [
    {"time": 0.0, "stage": "cleanup", "model_weights": 0.0, "kv_paged_unused": 0.0, "kv_used": 0.0, "compression_buffers": 0.0, "total": 0.0},
    {"time": 2.0, "stage": "start_vllm", "model_weights": 2.32, "kv_paged_unused": 0.0, "kv_used": 0.0, "compression_buffers": 0.0, "total": 2.32},
    {"time": 15.0, "stage": "vllm_ready", "model_weights": 2.32, "kv_paged_unused": 8.02, "kv_used": 0.34, "compression_buffers": 0.0, "total": 10.68},
    {"time": 18.0, "stage": "send_request", "model_weights": 2.32, "kv_paged_unused": 8.02, "kv_used": 0.34, "compression_buffers": 0.0, "total": 10.68},
    {"time": 18.5, "stage": "during_copy", "model_weights": 2.32, "kv_paged_unused": 7.36, "kv_used": 1.00, "compression_buffers": 0.0, "total": 11.68},
    {"time": 25.0, "stage": "after_copy", "model_weights": 2.32, "kv_paged_unused": 8.02, "kv_used": 0.34, "compression_buffers": 0.0, "total": 10.68},
]

cachegen_data = [
    {"time": 0.0, "stage": "cleanup", "model_weights": 0.0, "kv_paged_unused": 0.0, "kv_used": 0.0, "compression_buffers": 0.0, "total": 0.0},
    {"time": 2.0, "stage": "start_vllm", "model_weights": 2.32, "kv_paged_unused": 0.0, "kv_used": 0.0, "compression_buffers": 0.0, "total": 2.32},
    {"time": 15.0, "stage": "vllm_ready", "model_weights": 2.32, "kv_paged_unused": 8.02, "kv_used": 0.34, "compression_buffers": 0.0, "total": 10.68},
    {"time": 18.0, "stage": "send_request", "model_weights": 2.32, "kv_paged_unused": 8.02, "kv_used": 0.34, "compression_buffers": 0.0, "total": 10.68},
    {"time": 18.5, "stage": "during_compression", "model_weights": 2.32, "kv_paged_unused": 7.36, "kv_used": 1.00, "compression_buffers": 0.67, "total": 12.35},
    {"time": 25.0, "stage": "after_compression", "model_weights": 2.32, "kv_paged_unused": 8.02, "kv_used": 0.34, "compression_buffers": 0.0, "total": 10.68},
]

def create_graph(data, filename, title):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    times = [d["time"] for d in data]
    
    model = [d["model_weights"] for d in data]
    kv_paged = [d["kv_paged_unused"] for d in data]
    kv_used = [d["kv_used"] for d in data]
    comp_buffers = [d["compression_buffers"] for d in data]
    
    ax.stackplot(times, model, kv_paged, kv_used, comp_buffers,
                 labels=['Model Weights', 'KV Cache (Paged/Unused)', 'KV Cache (Used)', 'Compression Buffers'],
                 colors=['#3498db', '#27ae60', '#2ecc71', '#e74c3c'], alpha=0.85)
    
    totals = [d["total"] for d in data]
    ax.plot(times, totals, 'ko-', markersize=10, linewidth=2, label='Total VRAM (nvidia-smi)')
    
    for t, v in zip(times, totals):
        ax.annotate(f'{v:.1f}GB', xy=(t, v + 0.2), fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('VRAM Usage (GB)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 30)
    
    plt.tight_layout()
    output_path = f"{RESULT_DIR}/{filename}"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def verify(data):
    print(f"\nVerification:")
    print(f"{'Stage':<20} {'Total':<10} {'Sum':<10} {'Match'}")
    print("-" * 50)
    for d in data:
        total = d["total"]
        sum_v = d["model_weights"] + d["kv_paged_unused"] + d["kv_used"] + d["compression_buffers"]
        match = "OK" if abs(total - sum_v) < 0.1 else "FAIL"
        print(f"{d['stage']:<20} {total:<10.2f} {sum_v:<10.2f} {match}")

print("Creating Native breakdown graph...")
create_graph(native_data, "vram_breakdown_native_correct.png", "Native Mode VRAM Breakdown\n(KV Copy Only - No Compression)")
verify(native_data)

print("\nCreating CacheGen breakdown graph...")
create_graph(cachegen_data, "vram_breakdown_cachegen_correct.png", "CacheGen Mode VRAM Breakdown\n(KV Compression with PagedAttention)")
verify(cachegen_data)

print("\nDone!")
