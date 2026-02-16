#!/usr/bin/env python3
"""VRAM Layout Verification Test"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_sweep_experiment import FullVRAMMonitor, VRAMSnapshot
import time

def test_vram_monitor():
    print("="*60)
    print("VRAM Layout Verification Test")
    print("="*60)
    
    monitor = FullVRAMMonitor(port=8000, is_cachegen=False)
    
    print("\n[Test 1] VRAM Measurement (No vLLM)")
    snapshot = monitor.measure(is_cachegen=False)
    snapshot.print_layout("Initial State (No vLLM)")
    
    print("\n[Test 2] VRAM Layout Dict")
    layout = snapshot.to_layout_dict()
    print(f"Layout keys: {list(layout.keys())}")
    for k, v in layout.items():
        print(f"  {k}: {v:.4f} GB")
    
    print("\n[Test 3] Sum Validation")
    print(f"  nvidia-smi used_vram: {snapshot.used_vram_gb:.4f} GB")
    print(f"  sum_regions:          {snapshot.sum_validated_gb:.4f} GB")
    print(f"  sum_diff:             {snapshot.sum_diff_gb:+.4f} GB")
    
    print("\n[Test 4] VRAMSnapshot Fields")
    fields = [f for f in dir(snapshot) if not f.startswith('_')]
    print(f"  Total fields: {len(fields)}")
    for f in fields[:15]:
        print(f"    {f}")
    
    print("\n" + "="*60)
    print("Verification Complete")
    print("="*60)

if __name__ == "__main__":
    test_vram_monitor()
