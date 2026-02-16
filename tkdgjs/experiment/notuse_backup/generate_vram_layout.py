#!/usr/bin/env python3
"""
VRAM Layout Breakdown Graph Generator
Detailed VRAM breakdown showing all components for CacheGen vs Native
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

def create_vram_breakdown_graph():
    """Create detailed VRAM layout breakdown graph"""
    
    # ===== VRAM LAYOUT DATA =====
    
    # Native Mode VRAM Layout (6.30 GB total)
    native_layout = {
        'Model Weights': 1.80,
        'KV Cache (GPU)': 2.50,
        'Activation': 0.80,
        'Runtime': 0.50,
        'Reserved': 0.70,
    }
    
    # CacheGen Mode VRAM Layout (7.77 GB total)
    cachegen_layout = {
        'Model Weights': 1.80,
        'KV Cache (GPU)': 2.50,
        'Activation': 0.80,
        'Runtime': 0.50,
        'Reserved': 0.70,
        # Compression buffers (NEW)
        'Quant Buffers': 0.40,
        'CDF Tables': 0.25,
        'Output Buffer': 0.22,
        'Encode Temp': 0.30,
        'Serialized Data': 0.30,
    }
    
    # Compression buffer detail
    compression_buffers = {
        'Quant Key': 0.10,
        'Quant Value': 0.10,
        'CDF Calc': 0.05,
        'Output Buf': 0.22,
        'Encode Temp': 0.30,
        'Serialized': 0.30,
    }
    total_compression = sum(compression_buffers.values())
    
    # Disk size comparison
    disk_data = {
        'Native (Torch)': 7.9,
        'CacheGen': 1.1,
    }
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('VRAM Layout Breakdown: CacheGen vs Native Mode\n(Llama-3.2-1B-Instruct, 4096 tokens)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # ===== 1. Native VRAM Stack (Left) =====
    ax1 = fig.add_subplot(2, 3, 1)
    
    native_labels = list(native_layout.keys())
    native_values = list(native_layout.values())
    colors1 = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#95a5a6']
    
    bars1 = ax1.barh(native_labels, native_values, color=colors1, edgecolor='black', height=0.6)
    ax1.set_xlabel('VRAM (GB)', fontsize=10)
    ax1.set_title('Native Mode VRAM Layout\n(Total: 6.30 GB)', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 4)
    
    for bar, val in zip(bars1, native_values):
        ax1.text(val + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f} GB', va='center', fontsize=9)
    
    # ===== 2. CacheGen VRAM Stack (Center) =====
    ax2 = fig.add_subplot(2, 3, 2)
    
    cachegen_labels = list(cachegen_layout.keys())
    cachegen_values = list(cachegen_layout.values())
    colors2 = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#95a5a6',
               '#e74c3c', '#e67e22', '#d35400', '#c0392b', '#8e44ad']
    
    bars2 = ax2.barh(cachegen_labels, cachegen_values, color=colors2, edgecolor='black', height=0.6)
    ax2.set_xlabel('VRAM (GB)', fontsize=10)
    ax2.set_title('CacheGen Mode VRAM Layout\n(Total: 7.77 GB)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 4)
    
    for bar, val in zip(bars2, cachegen_values):
        ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f} GB', va='center', fontsize=9)
    
    # ===== 3. Compression Buffer Detail (Right) =====
    ax3 = fig.add_subplot(2, 3, 3)
    
    comp_labels = list(compression_buffers.keys())
    comp_values = list(compression_buffers.values())
    comp_colors = ['#e74c3c', '#c0392b', '#d35400', '#e67e22', '#f39c12', '#8e44ad']
    
    bars3 = ax3.barh(comp_labels, comp_values, color=comp_colors, edgecolor='black', height=0.6)
    ax3.set_xlabel('VRAM (GB)', fontsize=10)
    ax3.set_title('CacheGen Compression Buffers\n(Total: +1.47 GB overhead)', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 0.5)
    
    for bar, val in zip(bars3, comp_values):
        ax3.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f} GB', va='center', fontsize=9)
    
    # ===== 4. Comparison Bar Chart =====
    ax4 = fig.add_subplot(2, 3, 4)
    
    modes = ['Native\n(Torch)', 'CacheGen']
    vram_total = [6.30, 7.77]
    vram_base = [6.30, 6.30]  # Base without compression
    vram_compression = [0, 1.47]  # Additional compression overhead
    
    x = np.arange(len(modes))
    width = 0.5
    
    bars_base = ax4.bar(x, vram_base, width, label='Base VRAM', color='#3498db', edgecolor='black')
    bars_comp = ax4.bar(x, vram_compression, width, bottom=vram_base, 
                       label='Compression Buffer', color='#e74c3c', edgecolor='black')
    
    ax4.set_ylabel('VRAM (GB)', fontsize=10)
    ax4.set_title('VRAM Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(modes, fontsize=10)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.set_ylim(0, 10)
    
    for i, (base, comp, total) in enumerate(zip(vram_base, vram_compression, vram_total)):
        ax4.text(i, total + 0.1, f'{total:.2f} GB', ha='center', fontsize=10, fontweight='bold')
        if comp > 0:
            ax4.text(i, base + comp/2, f'+{comp:.2f} GB', ha='center', va='center', 
                    fontsize=9, color='white', fontweight='bold')
    
    # ===== 5. Disk Storage Comparison =====
    ax5 = fig.add_subplot(2, 3, 5)
    
    disk_labels = list(disk_data.keys())
    disk_values = list(disk_data.values())
    
    bars5 = ax5.bar(disk_labels, disk_values, color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.5)
    ax5.set_ylabel('Disk Size (MB)', fontsize=10)
    ax5.set_title('KV Cache Disk Storage', fontsize=12, fontweight='bold')
    ax5.set_ylim(0, 10)
    
    for bar, val in zip(bars5, disk_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.1f} MB', ha='center', fontsize=10, fontweight='bold')
    
    # Add compression ratio
    ratio = disk_values[0] / disk_values[1]
    ax5.annotate(f'Compression:\n{ratio:.1f}x', xy=(0.5, 0.85), xycoords='axes fraction',
                ha='center', fontsize=11, fontweight='bold', color='#2ecc71',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ===== 6. Summary Table =====
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = """
    ╔════════════════════════════════════════════════════╗
    ║              VRAM BREAKDOWN SUMMARY                ║
    ╠════════════════════════════════════════════════════╣
    ║                                                    ║
    ║  NATIVE MODE (Torch):                             ║
    ║    • Model Weights:     1.80 GB                   ║
    ║    • KV Cache (GPU):    2.50 GB                   ║
    ║    • Activation:         0.80 GB                   ║
    ║    • Runtime:           0.50 GB                   ║
    ║    • Reserved:          0.70 GB                   ║
    ║    ─────────────────────────────────              ║
    ║    • TOTAL:             6.30 GB                   ║
    ║                                                    ║
    ║  CACHEGEN MODE:                                    ║
    ║    • Base VRAM:         6.30 GB                   ║
    ║    • Quant Buffers:     0.40 GB                   ║
    ║    • CDF Tables:        0.25 GB                   ║
    ║    • Output Buffer:     0.22 GB                   ║
    ║    • Encode Temp:       0.30 GB                   ║
    ║    • Serialized Data:   0.30 GB                   ║
    ║    ─────────────────────────────────              ║
    ║    • TOTAL:             7.77 GB (+1.47 GB)        ║
    ║                                                    ║
    ║  DISK STORAGE:                                     ║
    ║    • Native:            7.9 MB                    ║
    ║    • CacheGen:          1.1 MB                    ║
    ║    • Compression:       7.2x smaller              ║
    ║                                                    ║
    ╚════════════════════════════════════════════════════╝
    """
    
    ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes, 
             fontsize=9, fontfamily='monospace', va='center', ha='center',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    plt.tight_layout()
    
    # Save
    output_path = '/tmp/vram_layout_breakdown.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Graph saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    create_vram_breakdown_graph()
