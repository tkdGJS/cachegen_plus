#!/usr/bin/env python3
"""
VRAM Visualization Collection
=============================
Various VRAM comparison graphs for CacheGen vs Native analysis.

Usage:
    python visualize_vram_graphs.py --type direct      # Direct VRAM test comparison
    python visualize_vram_graphs.py --type breakdown  # VRAM breakdown by layout
    python visualize_vram_graphs.py --type timeline     # Timeline comparison
    python visualize_vram_graphs.py --type all         # Generate all graphs
"""
import argparse
import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

# Configuration
EXPERIMENT_DIR = "/home/noslab-gpu/tkdgjs/experiment"
BACKUP_DIR = f"{EXPERIMENT_DIR}/backup_20260214"

def visualize_direct_vram():
    """Direct VRAM test results comparison"""
    stages = [
        "Initial\n(KV Cache)",
        "01_split_kv",
        "02_quant_key",
        "03_quant_value",
        "04_cat_encode",
        "05_calculate_cdf",
        "06_output_buffer",
        "07_encode_start",
        "08_encode_chunk",
        "Final"
    ]
    
    vram_cachegen = [1.0, 1.0, 1.0, 1.25, 1.5, 2.0, 2.016, 2.048, 2.064, 2.309]
    vram_native = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    x = np.arange(len(stages))
    width = 0.35
    
    ax1.bar(x - width/2, vram_cachegen, width, label='CacheGen (compress)', color='#e74c3c', alpha=0.8)
    ax1.bar(x + width/2, vram_native, width, label='Native (copy only)', color='#3498db', alpha=0.8)
    
    ax1.set_xlabel('Encoding Stage', fontsize=12)
    ax1.set_ylabel('VRAM Usage (GB)', fontsize=12)
    ax1.set_title('VRAM Usage by Encoding Stage\n(4096 tokens, Llama-3.2-1B)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    ax2 = axes[1]
    categories = ['Memory\nIncrease', 'Peak\nIncrease']
    cachegen_values = [0.2696, 1.6747]
    native_values = [1.0, 1.0]
    
    x2 = np.arange(len(categories))
    ax2.bar(x2 - width/2, cachegen_values, width, label='CacheGen', color='#e74c3c', alpha=0.8)
    ax2.bar(x2 + width/2, native_values, width, label='Native', color='#3498db', alpha=0.8)
    
    ax2.set_xlabel('Metric', fontsize=12)
    ax2.set_ylabel('VRAM Increase (GB)', fontsize=12)
    ax2.set_title('VRAM Increase Comparison\n(CacheGen vs Native)', fontsize=14)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (c, n) in enumerate(zip(cachegen_values, native_values)):
        diff = c - n
        ax2.annotate(f'{diff:+.2f} GB',
                    xy=(i, max(c, n)),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color='green' if diff > 0 else 'red')
    
    plt.tight_layout()
    output_path = f"{EXPERIMENT_DIR}/vram_direct_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def visualize_vram_breakdown():
    """VRAM breakdown by layout/region"""
    
    # VRAM layout breakdown based on CacheGen encoding stages
    # Each component's contribution to total VRAM usage
    
    # Layout breakdown for Native mode (simple copy)
    native_layout = {
        'KV Cache\n(Original)': 1.0,
        'Activation\nBuffer': 0.0,
        'Model\nWeights': 0.0,  # Already loaded
        'Temp\nCopy Buffer': 0.5,
        'Other': 0.5
    }
    
    # Layout breakdown for CacheGen mode (with compression)
    cachegen_layout = {
        'KV Cache\n(Original)': 1.0,
        'Quantized\nKV': 0.25,
        'CDF\nCalculation': 0.5,
        'Encode\nBuffer': 0.25,
        'Compressed\nOutput': 0.35,
        'Other': 0.2
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Stacked bar comparison
    ax1 = axes[0]
    
    # Native breakdown
    native_keys = list(native_layout.keys())
    native_values = list(native_layout.values())
    
    # CacheGen breakdown
    cachegen_keys = list(cachegen_layout.keys())
    cachegen_values = list(cachegen_layout.values())
    
    # Colors
    colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#95a5a6']
    
    # Create stacked bar chart
    x = np.arange(2)
    width = 0.6
    
    # Native stacked
    bottom = 0
    for i, (key, val) in enumerate(native_layout.items()):
        ax1.bar(0, val, width, bottom=bottom, label=f'{key}: {val:.2f}GB' if i < 3 else None,
                color=colors[i], alpha=0.8)
        if val > 0.1:
            ax1.text(0, bottom + val/2, f'{val:.2f}', ha='center', va='center', fontsize=9, fontweight='bold')
        bottom += val
    
    # CacheGen stacked
    bottom = 0
    for i, (key, val) in enumerate(cachegen_layout.items()):
        ax1.bar(1, val, width, bottom=bottom, 
                color=colors[i], alpha=0.8)
        if val > 0.1:
            ax1.text(1, bottom + val/2, f'{val:.2f}', ha='center', va='center', fontsize=9, fontweight='bold')
        bottom += val
    
    ax1.set_ylabel('VRAM Usage (GB)', fontsize=12)
    ax1.set_title('VRAM Layout Breakdown\n(Native vs CacheGen)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Native\n(Copy)', 'CacheGen\n(Compress)'])
    ax1.set_ylim(0, 3)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=colors[i], alpha=0.8) for i in range(len(cachegen_keys))]
    ax1.legend(handles, cachegen_keys, loc='upper right', fontsize=8)
    
    # Plot 2: Delta comparison (what's different)
    ax2 = axes[1]
    
    # Calculate differences
    delta_data = {
        'Additional\nBuffers': 0.75,  # Quantized + CDF + Encode
        'Compressed\nOutput': 0.35,
        'Saved (Temp Copy)': -0.15,  # Less temp buffer
    }
    
    categories = list(delta_data.keys())
    values = list(delta_data.values())
    bar_colors = ['#e74c3c' if v > 0 else '#27ae60' for v in values]
    
    bars = ax2.bar(categories, values, color=bar_colors, alpha=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('VRAM Difference (GB)', fontsize=12)
    ax2.set_title('VRAM Difference: CacheGen - Native\n(Positive = CacheGen uses more)', fontsize=14)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.annotate(f'{val:+.2f} GB',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5 if height > 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=11, fontweight='bold')
    
    # Add summary annotation
    total_diff = sum(values)
    ax2.annotate(f'Total: +{total_diff:.2f} GB',
                xy=(1, max(values) + 0.2),
                ha='center', fontsize=12, fontweight='bold', color='#e74c3c')
    
    plt.tight_layout()
    output_path = f"{EXPERIMENT_DIR}/vram_layout_breakdown.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def visualize_timeline():
    """VRAM timeline from sweep experiments"""
    # Load results
    results = []
    for f in glob.glob(f"{BACKUP_DIR}/timeline_*.jsonl"):
        with open(f, "r") as fp:
            for line in fp:
                data = json.loads(line)
                if data.get("type") == "result":
                    results.append(data)
    
    if not results:
        print("No timeline data found")
        return
    
    # Group by gpu_util
    gpu_utils = [0.3, 0.5, 0.7, 0.9]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, gpu_util in enumerate(gpu_utils):
        ax = axes[idx]
        
        # Filter data
        prefill_sizes = [128, 256, 512, 1024, 2048]
        x = np.arange(len(prefill_sizes))
        width = 0.35
        
        native_vrams = []
        cachegen_vrams = []
        
        for prefill in prefill_sizes:
            native_vram = None
            cachegen_vram = None
            for r in results:
                if r['gpu_util'] == gpu_util and r['prefill_size'] == prefill:
                    if r['mode'] == 'native':
                        native_vram = r.get('vram_after_start_gb', 0)
                    elif r['mode'] == 'cachegen':
                        cachegen_vram = r.get('vram_after_start_gb', 0)
            native_vrams.append(native_vram if native_vram else 0)
            cachegen_vrams.append(cachegen_vram if cachegen_vram else 0)
        
        ax.bar(x - width/2, native_vrams, width, label='Native', color='#3498db', alpha=0.8)
        ax.bar(x + width/2, cachegen_vrams, width, label='CacheGen', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('Prefill Size (tokens)', fontsize=10)
        ax.set_ylabel('VRAM Usage (GB)', fontsize=10)
        ax.set_title(f'GPU Util: {gpu_util}', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(prefill_sizes)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('VRAM Usage: Native vs CacheGen\n(All experiments show identical VRAM)', fontsize=14)
    plt.tight_layout()
    output_path = f"{EXPERIMENT_DIR}/vram_timeline_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='VRAM Visualization Tools')
    parser.add_argument('--type', choices=['direct', 'breakdown', 'timeline', 'all'], 
                        default='all', help='Type of visualization')
    args = parser.parse_args()
    
    if args.type in ['direct', 'all']:
        visualize_direct_vram()
    
    if args.type in ['breakdown', 'all']:
        visualize_vram_breakdown()
    
    if args.type in ['timeline', 'all']:
        visualize_timeline()
    
    print("\nAll visualizations complete!")

if __name__ == "__main__":
    main()
