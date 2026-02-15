#!/usr/bin/env python3
"""
VRAM Timeline Visualization
===========================
Shows VRAM usage across different GPU utilization levels and prefill sizes.
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import glob

BACKUP_DIR = "/home/noslab-gpu/tkdgjs/experiment/backup_20260214"

# Load results
results = []
for f in glob.glob(f"{BACKUP_DIR}/timeline_*.jsonl"):
    with open(f, "r") as fp:
        for line in fp:
            data = json.loads(line)
            if data.get("type") == "result":
                results.append(data)

print(f"Loaded {len(results)} results")

gpu_utils = [0.3, 0.5, 0.7, 0.9]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, gpu_util in enumerate(gpu_utils):
    ax = axes[idx]
    
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
    
    ax.bar(x - width/2, native_vrams, width, label='Native', color='#3498db', alpha=0.85)
    ax.bar(x + width/2, cachegen_vrams, width, label='CacheGen', color='#e74c3c', alpha=0.85)
    
    ax.set_xlabel('Prefill Size (tokens)', fontsize=10)
    ax.set_ylabel('VRAM Usage (GB)', fontsize=10)
    ax.set_title(f'GPU Util: {gpu_util}', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(prefill_sizes)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('VRAM Usage: Native vs CacheGen by GPU Utilization\n(All experiments show identical VRAM - compression buffers not captured in nvidia-smi)', 
             fontsize=14)
plt.tight_layout()
output_path = '/home/noslab-gpu/tkdgjs/experiment/result/vram_timeline_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\nSaved: {output_path}")
print("\nNote: Native and CacheGen show identical VRAM in sweep tests because")
print("nvidia-smi doesn't capture the temporary compression buffers.")
print("Use direct VRAM test for accurate compression overhead measurement.")
