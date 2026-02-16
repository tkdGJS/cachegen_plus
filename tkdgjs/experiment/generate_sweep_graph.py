#!/usr/bin/env python3
"""
Generate sweep experiment results graph
"""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os

def load_results():
    """Load sweep results"""
    try:
        with open('/tmp/sweep_results.json', 'r') as f:
            return json.load(f)
    except:
        return []

def create_sweep_graph(results):
    """Create sweep experiment graphs"""
    
    # Separate by mode
    cachegen = [r for r in results if r.get('mode') == 'cachegen' and r.get('status') == 'success']
    native = [r for r in results if r.get('mode') == 'native' and r.get('status') == 'success']
    
    if not cachegen and not native:
        print("No results found!")
        return None
    
    # Create figure
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('LMCache VRAM Sweep Experiment Results\n(CacheGen vs Native Mode)', 
                fontsize=16, fontweight='bold')
    
    # Prepare data
    tokens_list = sorted(list(set([r['tokens'] for r in cachegen + native])))
    gpu_mem_list = sorted(list(set([r['gpu_memory_utilization'] for r in cachegen + native])))
    
    # 1. VRAM vs Tokens (for each GPU mem)
    for i, gpu_mem in enumerate(gpu_mem_list):
        ax = fig.add_subplot(2, 3, i+1)
        
        c_data = [r for r in cachegen if r['gpu_memory_utilization'] == gpu_mem]
        n_data = [r for r in native if r['gpu_memory_utilization'] == gpu_mem]
        
        c_tokens = [r['tokens'] for r in c_data]
        c_vram = [r.get('vram_before_gb', 0) for r in c_data]
        
        n_tokens = [r['tokens'] for r in n_data]
        n_vram = [r.get('vram_before_gb', 0) for r in n_data]
        
        if c_tokens:
            ax.plot(c_tokens, c_vram, 'o-', color='red', label='CacheGen', linewidth=2, markersize=8)
        if n_tokens:
            ax.plot(n_tokens, n_vram, 's-', color='green', label='Native', linewidth=2, markersize=8)
        
        ax.set_xlabel('Tokens', fontsize=10)
        ax.set_ylabel('VRAM (GB)', fontsize=10)
        ax.set_title(f'GPU Memory: {gpu_mem}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 4. Disk Size Comparison
    ax4 = fig.add_subplot(2, 3, 4)
    
    c_disk = [r.get('disk_size_mb', 0) for r in cachegen if r.get('disk_size_mb', 0) > 0]
    n_disk = [r.get('disk_size_mb', 0) for r in native if r.get('disk_size_mb', 0) > 0]
    
    x = np.arange(len(cachegen))
    width = 0.35
    
    if c_disk and n_disk:
        ax4.bar(x - width/2, c_disk, width, label='CacheGen', color='red', alpha=0.7)
        ax4.bar(x + width/2, n_disk, width, label='Native', color='green', alpha=0.7)
    
    ax4.set_xlabel('Test Index', fontsize=10)
    ax4.set_ylabel('Disk Size (MB)', fontsize=10)
    ax4.set_title('KV Cache Disk Size', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. VRAM Difference (CacheGen - Native)
    ax5 = fig.add_subplot(2, 3, 5)
    
    diffs = []
    labels = []
    for r in cachegen:
        tokens = r['tokens']
        gpu_mem = r['gpu_memory_utilization']
        # Find matching native result
        n = [x for x in native if x['tokens'] == tokens and x['gpu_memory_utilization'] == gpu_mem]
        if n:
            diff = r.get('vram_before_gb', 0) - n[0].get('vram_before_gb', 0)
            diffs.append(diff)
            labels.append(f"{tokens}/{gpu_mem}")
    
    if diffs:
        colors = ['red' if d > 0 else 'green' for d in diffs]
        ax5.bar(range(len(diffs)), diffs, color=colors, alpha=0.7)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax5.set_xticks(range(len(diffs)))
        ax5.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    ax5.set_xlabel('Tokens/GPU-Mem', fontsize=10)
    ax5.set_ylabel('VRAM Difference (GB)', fontsize=10)
    ax5.set_title('VRAM Diff (CacheGen - Native)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary Table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate stats
    avg_vram_cachegen = np.mean([r.get('vram_before_gb', 0) for r in cachegen])
    avg_vram_native = np.mean([r.get('vram_before_gb', 0) for r in native])
    avg_disk_cachegen = np.mean([r.get('disk_size_mb', 0) for r in cachegen if r.get('disk_size_mb', 0) > 0])
    avg_disk_native = np.mean([r.get('disk_size_mb', 0) for r in native if r.get('disk_size_mb', 0) > 0])
    
    if avg_disk_native > 0:
        compression_ratio = avg_disk_native / avg_disk_cachegen
    else:
        compression_ratio = 0
    
    summary = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    SWEEP EXPERIMENT SUMMARY                           ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Total Experiments: {len(results)}                                              ║
║  Successful: {len([r for r in results if r.get('status') == 'success'])}                                              ║
║  Failed: {len([r for r in results if r.get('status') != 'success'])}                                               ║
║                                                                    ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  AVERAGE VRAM USAGE:                                               ║
║    CacheGen: {avg_vram_cachegen:.2f} GB                                         ║
║    Native:   {avg_vram_native:.2f} GB                                         ║
║    Diff:     {avg_vram_cachegen - avg_vram_native:+.2f} GB                                          ║
║                                                                    ║
║  AVERAGE DISK SIZE:                                               ║
║    CacheGen: {avg_disk_cachegen:.2f} MB                                         ║
║    Native:   {avg_disk_native:.2f} MB                                         ║
║    Compression: {compression_ratio:.1f}x                                           ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
    
    ax6.text(0.5, 0.5, summary, transform=ax6.transAxes,
            fontsize=9, fontfamily='monospace', va='center', ha='center',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa'))
    
    plt.tight_layout()
    
    output_path = '/tmp/sweep_results_graph.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Graph saved to: {output_path}")
    
    # Also save summary
    with open('/tmp/sweep_summary.txt', 'w') as f:
        f.write(f"""SWEEP EXPERIMENT RESULTS
========================

Total Experiments: {len(results)}
Successful: {len([r for r in results if r.get('status') == 'success'])}
Failed: {len([r for r in results if r.get('status') != 'success'])}

Average VRAM:
  CacheGen: {avg_vram_cachegen:.2f} GB
  Native: {avg_vram_native:.2f} GB
  Difference: {avg_vram_cachegen - avg_vram_native:+.2f} GB

Average Disk Size:
  CacheGen: {avg_disk_cachegen:.2f} MB
  Native: {avg_disk_native:.2f} MB
  Compression: {compression_ratio:.1f}x
""")
    
    return output_path

if __name__ == "__main__":
    results = load_results()
    if results:
        create_sweep_graph(results)
    else:
        print("No results to plot. Run sweep experiment first.")
