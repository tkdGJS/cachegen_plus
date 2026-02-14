#!/usr/bin/env python3
"""VRAM Layout Verification with GPU Cleanup"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_sweep_experiment import clear_gpu_processes, FullVRAMMonitor
import subprocess

def verify():
    print("="*60)
    print("VRAM Layout Verification with GPU Cleanup")
    print("="*60)
    
    # Step 1: Clean GPU
    print("\n[Step 1] GPU Cleanup")
    used_before = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True
    ).stdout.strip()
    print(f"  Before cleanup: {used_before} MB")
    
    used_after = clear_gpu_processes()
    print(f"  After cleanup: {used_after:.2f} MB")
    
    # Step 2: Check idle state
    print("\n[Step 2] Idle GPU State")
    monitor = FullVRAMMonitor(port=8000, is_cachegen=False)
    snapshot = monitor.measure(is_cachegen=False)
    snapshot.print_layout("GPU Idle (After Cleanup)")
    
    idle_match = abs(snapshot.used_vram_gb - snapshot.sum_validated_gb) < 0.5
    print(f"\n  Idle sum match: {idle_match}")
    
    # Step 3: Verify layout fields
    print("\n[Step 3] VRAM Layout Fields")
    layout = snapshot.to_layout_dict()
    print(f"  Fields: {list(layout.keys())}")
    
    # Step 4: Check reserved memory
    print("\n[Step 4] Reserved Memory")
    print(f"  Reserved (nvidia-smi): {snapshot.reserved_vram_gb:.4f} GB")
    print(f"  Used: {snapshot.used_vram_gb:.4f} GB")
    print(f"  Free: {snapshot.free_vram_gb:.4f} GB")
    
    print("\n" + "="*60)
    print("Verification Complete")
    print("="*60)
    
    return {
        "cleanup_working": used_after < 1.0,
        "idle_match": idle_match,
        "reserved_tracked": snapshot.reserved_vram_gb > 0
    }

if __name__ == "__main__":
    result = verify()
    print(f"\nResults: {result}")
