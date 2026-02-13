#!/usr/bin/env python3
"""
VRAM Monitor - 0.1초 간격으로 VRAM 사용량 측정
"""
import subprocess
import time
import threading
import json
import sys
import os

class VRAMMonitor:
    def __init__(self, interval=0.1, output_file="vram_log.jsonl"):
        self.interval = interval
        self.output_file = output_file
        self.running = False
        self.samples = []
        self.start_time = None
        
    def get_vram(self):
        """nvidia-smi로 VRAM 사용량 조회 (MB 단위)"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", 
                 "--format=csv,noheader,nounits", "-i", "0"],
                capture_output=True, text=True, timeout=5
            )
            return int(result.stdout.strip())
        except Exception as e:
            print(f"Error getting VRAM: {e}", file=sys.stderr)
            return None
    
    def monitor_loop(self):
        while self.running:
            vram = self.get_vram()
            if vram is not None:
                elapsed = time.time() - self.start_time
                self.samples.append({
                    "elapsed_sec": round(elapsed, 3),
                    "timestamp": time.time(),
                    "vram_mb": vram,
                    "vram_gb": round(vram / 1024, 2)
                })
            time.sleep(self.interval)
    
    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.thread.start()
        print(f"[VRAMMonitor] Started monitoring to {self.output_file}")
        
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        print(f"[VRAMMonitor] Stopped. Collected {len(self.samples)} samples")
        
    def save(self):
        with open(self.output_file, "w") as f:
            json.dump(self.samples, f, indent=2)
        print(f"[VRAMMonitor] Saved to {self.output_file}")
        
    def get_stats(self):
        if not self.samples:
            return {}
        vram_values = [s["vram_mb"] for s in self.samples]
        return {
            "min_mb": min(vram_values),
            "max_mb": max(vram_values),
            "peak_gb": max(vram_values) / 1024,
            "sample_count": len(vram_values)
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="vram_log.jsonl")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration in seconds")
    args = parser.parse_args()
    
    monitor = VRAMMonitor(interval=0.1, output_file=args.output)
    monitor.start()
    time.sleep(args.duration)
    monitor.stop()
    monitor.save()
    
    stats = monitor.get_stats()
    print(f"Stats: {stats}")
