#!/usr/bin/env python3
"""
VRAM Measurement Test - Native vs CacheGen
Using LMCACHE_VRAM_LOG=1 to measure compression VRAM
"""
import subprocess
import time
import os
import sys

VLLM_BIN = "/home/noslab-gpu/tkdgjs/tkdgjs/bin/vllm"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
PORT = 8000

def cleanup():
    subprocess.run(["pkill", "-9", "-f", "vllm"], stderr=subprocess.DEVNULL)
    time.sleep(3)

def start_vllm(mode: str, gpu_mem: float = 0.3):
    """Start vLLM with specified mode"""
    config_file = f"/home/noslab-gpu/tkdgjs/experiment/lmcache_{mode}.yaml"
    
    env = os.environ.copy()
    env["LMCACHE_CONFIG_FILE"] = config_file
    env["LMCACHE_VRAM_LOG"] = "1"  # Enable VRAM logging
    env["PYTHONPATH"] = "/home/noslab-gpu/tkdgjs/tkdgjs/lib/python3.10/site-packages"
    
    cmd = [
        VLLM_BIN, "serve", MODEL,
        "--port", str(PORT),
        "--dtype", "half",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", str(gpu_mem),
        "--kv-transfer-config", '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}',
    ]
    
    log_file = f"/home/noslab-gpu/tkdgjs/experiment/vram_test_{mode}.log"
    proc = subprocess.Popen(cmd, env=env, stdout=open(log_file, "w"), stderr=subprocess.STDOUT)
    return proc, log_file

def wait_for_vllm(timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = subprocess.run(["curl", "-s", f"http://localhost:{PORT}/v1/models"],
                            capture_output=True, timeout=5)
            if r.returncode == 0:
                return True
        except:
            pass
        time.sleep(2)
    return False

def send_request(prompt_tokens: int = 2000):
    """Send a request and return success status"""
    prompt = "Hello world, this is a test. " * (prompt_tokens // 5)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 10,
        "temperature": 0.0
    }
    
    try:
        r = subprocess.run([
            "curl", "-s", "-X", "POST",
            f"http://localhost:{PORT}/v1/completions",
            "-H", "Content-Type: application/json",
            "-d", str(payload).replace("'", '"')
        ], capture_output=True, timeout=30)
        return r.returncode == 0
    except:
        return False

def extract_vram_logs(log_file: str):
    """Extract VRAM measurements from log file"""
    vram_logs = []
    try:
        with open(log_file, "r") as f:
            for line in f:
                if "[LMCACHE_VRAM]" in line:
                    # Parse: [LMCACHE_VRAM] encode_function: before=0.1234GB, after=0.5678GB, increase=0.4444GB
                    parts = line.split("[LMCACHE_VRAM]")[1].strip()
                    vram_logs.append(parts)
    except:
        pass
    return vram_logs

def run_test(mode: str):
    """Run test for a specific mode"""
    print(f"\n{'='*60}")
    print(f"Testing: {mode.upper()}")
    print(f"{'='*60}")
    
    # Cleanup
    cleanup()
    
    # Start vLLM
    print(f"Starting vLLM with {mode} mode...")
    proc, log_file = start_vllm(mode)
    
    # Wait for ready
    print("Waiting for vLLM to be ready...")
    if not wait_for_vllm():
        print(f"ERROR: vLLM not ready for {mode}")
        proc.kill()
        return None
    
    print(f"vLLM ready for {mode}!")
    
    # Send requests
    print("Sending requests...")
    for i in range(3):
        success = send_request(prompt_tokens=2000)
        print(f"  Request {i+1}: {'Success' if success else 'Failed'}")
        time.sleep(2)
    
    # Wait for logs to be written
    time.sleep(3)
    
    # Extract VRAM logs
    vram_logs = extract_vram_logs(log_file)
    
    # Cleanup
    proc.kill()
    
    print(f"\nVRAM Logs for {mode}:")
    for log in vram_logs:
        print(f"  {log}")
    
    return vram_logs

# Main test
print("="*60)
print("VRAM Measurement: Native vs CacheGen")
print("="*60)

# Test Native
native_logs = run_test("native")

time.sleep(10)

# Test CacheGen
cachegen_logs = run_test("cachegen")

# Compare results
print("\n" + "="*60)
print("COMPARISON RESULTS")
print("="*60)

print(f"\nNative VRAM logs: {len(native_logs) if native_logs else 0}")
print(f"CacheGen VRAM logs: {len(cachegen_logs) if cachegen_logs else 0}")

if native_logs and cachegen_logs:
    # Parse and compare
    def parse_vram(logs):
        increases = []
        for log in logs:
            if "increase=" in log:
                # Extract increase value
                for part in log.split(","):
                    if "increase=" in part:
                        val = part.split("=")[1].replace("GB", "").strip()
                        try:
                            increases.append(float(val))
                        except:
                            pass
        return increases
    
    native_increases = parse_vram(native_logs)
    cachegen_increases = parse_vram(cachegen_logs)
    
    print(f"\nNative VRAM increases: {native_increases}")
    print(f"CacheGen VRAM increases: {cachegen_increases}")
    
    if native_increases and cachegen_increases:
        avg_native = sum(native_increases) / len(native_increases)
        avg_cachegen = sum(cachegen_increases) / len(cachegen_increases)
        
        print(f"\nAverage VRAM increase:")
        print(f"  Native:   {avg_native:.4f} GB")
        print(f"  CacheGen: {avg_cachegen:.4f} GB")
        print(f"  Diff:     {avg_cachegen - avg_native:+.4f} GB")
else:
    print("\nERROR: Could not extract VRAM measurements")
    print("Check if LMCACHE_VRAM_LOG is working")

print("\n" + "="*60)
