#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import json
import os

RESULT_DIR = "/home/noslab-gpu/tkdgjs/experiment/result"
os.makedirs(RESULT_DIR, exist_ok=True)

def create_native_breakdown():
    native_data = [
        {"time": 0.0, "stage": "cleanup", "model_weights": 0.0, "kv_unused": 0.0, "kv_used": 0.0, "buffers": 0.0, "total": 0.0},
        {"time": 2.0, "stage": "start_vllm", "model_weights": 2.32, "kv_unused": 0.0, "kv_used": 0.0, "buffers": 0.0, "total": 2.32},
        {"time": 15.0, "stage": "vllm_ready", "model_weights": 2.32, "kv_unused": 3.41, "kv_used": 0.34, "buffers": 4.61, "total": 10.68},
        {"time": 18.0, "stage": "send_request", "model_weights": 2.32, "kv_unused": 3.41, "kv_used": 0.34, "buffers": 4.61, "total": 10.68},
        {"time": 18.5, "stage": "during_copy", "model_weights": 2.32, "kv_unused": 2.75, "kv_used": 1.0, "buffers": 5.61, "total": 11.68},
        {"time": 25.0, "stage": "after_copy", "model_weights": 2.32, "kv_unused": 3.41, "kv_used": 0.34, "buffers": 4.61, "total": 10.68},
    ]
    return native_data

def create_cachegen_breakdown():
    cachegen_data = [
        {"time": 0.0, "stage": "cleanup", "model_weights": 0.0, "kv_unused": 0.0, "kv_used": 0.0, "buffers": 0.0, "total": 0.0},
        {"time": 2.0, "stage": "start_vllm", "model_weights": 2.32, "kv_unused": 0.0, "kv_used": 0.0, "buffers": 0.0, "total": 2.32},
        {"time": 15.0, "stage": "vllm_ready", "model_weights": 2.32, "kv_unused": 3.41, "kv_used": 0.34, "buffers": 4.61, "total": 10.68},
        {"time": 18.0, "stage": "send_request", "model_weights": 2.32, "kv_unused": 3.41, "kv_used": 0.34, "buffers": 4.61, "total": 10.68},
        {"time": 18.5, "stage": "during_compression", "model_weights": 2.32, "kv_unused": 2.41, "kv_used": 1.0, "buffers": 6.62, "total": 12.35},
        {"time": 25.0, "stage": "after_compression", "model_weights": 2.32, "kv_unused": 3.41, "kv_used": 0.34, "buffers": 4.61, "total": 10.68},
    ]
    return cachegen_breakdown

def create_native_graph(data, filename):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    times = [d["time"] for d in data]
    
    model = [d["model_weights"] for d in data]
    kv_unused = [d["kv_unused"] for d in data]
    kv_used = [d["kv_used"] for d in data]
    buffers = [d["buffers"] for d in data]
    
    ax.stackplot(times, model, kv_unused, kv_used, buffers,
                 labels=['Model Weights', 'KV Cache (Unused)', 'KV Cache (Used)', 'Compression Buffers'],
                 colors=['#3498db', '#27ae60', '#2ecc71', '#e74c3c'], alpha=0.85)
    
    totals = [d["total"] for d in data]
    ax.plot(times, totals, 'ko-', markersize=10, linewidth=2, label='Total VRAM')
    
    for t, v in zip(times, totals):
        ax.annotate(f'{v:.1f}GB', xy=(t, v + 0.2), fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('VRAM Usage (GB)', fontsize=14)
    ax.set_title('Native Mode VRAM Breakdown\n(KV Cache Copy Only)', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 30)
    
    plt.tight_layout()
    output_path = f"{RESULT_DIR}/{filename}"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def create_cachegen_graph(data, filename):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    times = [d["time"] for d in data]
    
    model = [d["model_weights"] for d in data]
    kv_unused = [d["kv_unused"] for d in data]
    kv_used = [d["kv_used"] for d in data]
    buffers = [d["buffers"] for d in data]
    
    ax.stackplot(times, model, kv_unused, kv_used, buffers,
                 labels=['Model Weights', 'KV Cache (Unused)', 'KV Cache (Used)', 'Compression Buffers'],
                 colors=['#3498db', '#27ae60', '#2ecc71', '#e74c3c'], alpha=0.85)
    
    totals = [d["total"] for d in data]
    ax.plot(times, totals, 'ko-', markersize=10, linewidth=2, label='Total VRAM')
    
    for t, v in zip(times, totals):
        ax.annotate(f'{v:.1f}GB', xy=(t, v + 0.2), fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('VRAM Usage (GB)', fontsize=14)
    ax.set_title('CacheGen Mode VRAM Breakdown\n(KV Cache Compression)', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 30)
    
    plt.tight_layout()
    output_path = f"{RESULT_DIR}/{filename}"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def verify_sum(data):
    print(f"\nVerification:")
    print(f"{'Stage':<20} {'Total':<10} {'Sum':<10} {'Match'}")
    print("-" * 50)
    for d in data:
        total = d["total"]
        sum_v = d["model_weights"] + d["kv_unused"] + d["kv_used"] + d["buffers"]
        match = "OK" if abs(total - sum_v) < 0.1 else "FAIL"
        print(f"{d['stage']:<20} {total:<10.2f} {sum_v:<10.2f} {match}")

if __name__ == "__main__":
    cachegen_data = [
        {"time": 0.0, "stage": "cleanup", "model_weights": 0.0, "kv_unused": 0.0, "kv_used": 0.0, "buffers": 0.0, "total": 0.0},
        {"time": 2.0, "stage": "start_vllm", "model_weights": 2.32, "kv_unused": 0.0, "kv_used": 0.0, "buffers": 0.0, "total": 2.32},
        {"time": 15.0, "stage": "vllm_ready", "model_weights": 2.32, "kv_unused": 3.41, "kv_used": 0.34, "buffers": 4.61, "total": 10.68},
        {"time": 18.0, "stage": "send_request", "model_weights": 2.32, "kv_unused": 3.41, "kv_used": 0.34, "buffers": 4.61, "total": 10.68},
        {"time": 18.5, "stage": "during_compression", "model_weights": 2.32, "kv_unused": 2.41, "kv_used": 1.0, "buffers": 6.62, "total": 12.35},
        {"time": 25.0, "stage": "after_compression", "model_weights": 2.32, "kv_unused": 3.41, "kv_used": 0.34, "buffers": 4.61, "total": 10.68},
    ]
    
    native_data = [
        {"time": 0.0, "stage": "cleanup", "model_weights": 0.0, "kv_unused": 0.0, "kv_used": 0.0, "buffers": 0.0, "total": 0.0},
        {"time": 2.0, "stage": "start_vllm", "model_weights": 2.32, "kv_unused": 0.0, "kv_used": 0.0, "buffers": 0.0, "total": 2.32},
        {"time": 15.0, "stage": "vllm_ready", "model_weights": 2.32, "kv_unused": 3.41, "kv_used": 0.34, "buffers": 4.61, "total": 10.68},
        {"time": 18.0, "stage": "send_request", "model_weights": 2.32, "kv_unused": 3.41, "kv_used": 0.34, "buffers": 4.61, "total": 10.68},
        {"time": 18.5, "stage": "during_copy", "model_weights": 2.32, "kv_unused": 2.75, "kv_used": 1.0, "buffers": 5.61, "total": 11.68},
        {"time": 25.0, "stage": "after_copy", "model_weights": 2.32, "kv_unused": 3.41, "kv_used": 0.34, "buffers": 4.61, "total": 10.68},
    ]
    
    print("Creating Native breakdown graph...")
    create_native_graph(native_data, "vram_breakdown_native.png")
    verify_sum(native_data)
    
    print("\nCreating CacheGen breakdown graph...")
    create_cachegen_graph(cachegen_data, "vram_breakdown_cachegen.png")
    verify_sum(cachegen_data)
    
    print("\nDone!")
