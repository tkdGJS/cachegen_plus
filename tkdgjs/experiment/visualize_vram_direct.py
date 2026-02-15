#!/usr/bin/env python3
"""
Direct VRAM Test Results Visualization
- CacheGen vs Native VRAM usage comparison
- Shows detailed encoding stages
"""
import matplotlib.pyplot as plt
import numpy as np

# Direct VRAM test results (4096 tokens, Llama-3.2-1B)
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

# VRAM at each stage (GB) - from test output
vram_cachegen = [
    1.0,    # Initial KV cache
    1.0,    # 01_split_kv
    1.0,    # 02_quant_key
    1.25,   # 03_quant_value
    1.5,    # 04_cat_encode_input
    2.0,    # 05_calculate_cdf
    2.016,  # 06_output_buffer
    2.048,  # 07_encode_ntokens_start
    2.064,  # 08_encode_ntokens_chunk
    2.309   # Final output
]

# Native mode (simple copy)
vram_native = [
    1.0,    # Initial KV cache
    1.0,    # copy
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    2.0     # Final (1GB copy)
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Stage-by-stage VRAM comparison
ax1 = axes[0]
x = np.arange(len(stages))
width = 0.35

bars1 = ax1.bar(x - width/2, vram_cachegen, width, label='CacheGen (compress)', color='#e74c3c', alpha=0.8)
bars2 = ax1.bar(x + width/2, vram_native, width, label='Native (copy only)', color='#3498db', alpha=0.8)

ax1.set_xlabel('Encoding Stage', fontsize=12)
ax1.set_ylabel('VRAM Usage (GB)', fontsize=12)
ax1.set_title('VRAM Usage by Encoding Stage\n(4096 tokens, Llama-3.2-1B)', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(stages, rotation=45, ha='right', fontsize=8)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=7)

# Plot 2: Peak vs Memory increase comparison
ax2 = axes[1]

categories = ['Memory\nIncrease', 'Peak\nIncrease']
cachegen_values = [0.2696, 1.6747]
native_values = [1.0, 1.0]

x2 = np.arange(len(categories))
bars3 = ax2.bar(x2 - width/2, cachegen_values, width, label='CacheGen', color='#e74c3c', alpha=0.8)
bars4 = ax2.bar(x2 + width/2, native_values, width, label='Native', color='#3498db', alpha=0.8)

ax2.set_xlabel('Metric', fontsize=12)
ax2.set_ylabel('VRAM Increase (GB)', fontsize=12)
ax2.set_title('VRAM Increase Comparison\n(CacheGen vs Native)', fontsize=14)
ax2.set_xticks(x2)
ax2.set_xticklabels(categories)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add difference annotations
for i, (c, n) in enumerate(zip(cachegen_values, native_values)):
    diff = c - n
    ax2.annotate(f'{diff:+.2f} GB',
                xy=(i, max(c, n)),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color='green' if diff > 0 else 'red')

plt.tight_layout()
output_path = '/home/noslab-gpu/tkdgjs/experiment/result/vram_direct_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Graph saved to: /home/noslab-gpu/tkdgjs/experiment/vram_direct_comparison.png")

# Summary
print("\n" + "="*60)
print("SUMMARY: Direct VRAM Test Results")
print("="*60)
print(f"""
Configuration:
  - Model: Llama-3.2-1B-Instruct
  - Tokens: 4096
  - KV Cache Size: 1.0 GB

Results:
  ┌─────────────────────┬───────────────┬───────────────┐
  │ Metric              │ CacheGen      │ Native        │
  ├─────────────────────┼───────────────┼───────────────┤
  │ Memory Increase     │ 0.27 GB      │ 1.00 GB       │
  │ Peak Increase       │ 1.67 GB      │ 1.00 GB       │
  └─────────────────────┴───────────────┴───────────────┘

Conclusion:
  CacheGen uses +0.67 GB MORE peak VRAM than Native mode
  during the compression operation.
""")
