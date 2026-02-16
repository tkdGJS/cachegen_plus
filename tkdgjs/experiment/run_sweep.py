#!/usr/bin/env python3
"""
LMCache VRAM Sweep Experiment
Tests multiple configurations and handles OOM gracefully
"""

import os
import subprocess
import json
import time
import signal
from datetime import datetime

# Experiment configurations
CONFIGS = [
    # (tokens, gpu_memory_utilization)
    (128, 0.3),
    (256, 0.3),
    (512, 0.3),
    (1024, 0.3),
    (128, 0.5),
    (256, 0.5),
    (512, 0.5),
    (1024, 0.5),
    (2048, 0.5),
    (128, 0.7),
    (256, 0.7),
    (512, 0.7),
    (1024, 0.7),
]

# Alternative smaller configs for safety
SAFE_CONFIGS = [
    (128, 0.3),
    (256, 0.3),
    (512, 0.3),
    (1024, 0.3),
    (128, 0.5),
    (256, 0.5),
    (512, 0.5),
    (1024, 0.5),
    (2048, 0.5),
]

def get_base_cmd(tokens, gpu_mem):
    return [
        "/home/noslab-gpu/tkdgjs/tkdgjs/bin/vllm", "serve",
        "meta-llama/Llama-3.2-1B-Instruct",
        "--port", "8005",
        "--dtype", "half",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", str(gpu_mem),
        "--kv-transfer-config", '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"discard_partial_chunks":false}}'
    ]

def check_vllm_ready(port=8005, timeout=30):
    """Check if vLLM is ready"""
    for _ in range(timeout):
        try:
            result = subprocess.run(
                ["curl", "-s", f"http://localhost:{port}/v1/models"],
                capture_output=True, timeout=5
            )
            if result.returncode == 0 and b'"object":"list"' in result.stdout:
                return True
        except:
            pass
        time.sleep(1)
    return False

def kill_vllm():
    """Kill vLLM processes"""
    subprocess.run(["pkill", "-9", "-f", "vllm"], capture_output=True)
    time.sleep(3)

def check_gpu_memory():
    """Get free GPU memory in GB"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split('\n')[0]) / 1024.0
    except:
        pass
    return 0

def run_experiment(tokens, gpu_mem, mode, config_file):
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"Running: {mode} | Tokens: {tokens} | GPU: {gpu_mem}")
    print(f"{'='*60}")
    
    # Set environment
    env = os.environ.copy()
    env["LMCACHE_CONFIG_FILE"] = config_file
    env["LMCACHE_VRAM_LOG"] = "1"
    env["LMCACHE_VRAM_LOG_FILE"] = "/tmp/lmcache_vram.log"
    env["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
    
    # Clean cache directories
    if mode == "cachegen":
        cache_dir = "/home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen_disk"
    else:
        cache_dir = "/home/noslab-gpu/tkdgjs/experiment/lmcache_torch_disk"
    
    subprocess.run(["rm", "-rf", f"{cache_dir}/*"], shell=True, capture_output=True)
    subprocess.run(["rm", "-f", "/tmp/lmcache_vram.log"], capture_output=True)
    
    # Start vLLM
    cmd = get_base_cmd(tokens, gpu_mem)
    print(f"Starting vLLM...")
    
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for startup (max 90 seconds)
    startup_success = False
    for i in range(90):
        time.sleep(1)
        # Check if process is still running
        if proc.poll() is not None:
            # Process died
            stdout, stderr = proc.communicate()
            print(f"vLLM process died during startup")
            return {"status": "failed", "error": "process_died", "stdout": stdout.decode()[:500], "stderr": stderr.decode()[:500]}
        
        # Check if ready
        if check_vllm_ready():
            startup_success = True
            print(f"vLLM ready after {i+1} seconds")
            break
    
    if not startup_success:
        proc.kill()
        return {"status": "failed", "error": "startup_timeout"}
    
    time.sleep(2)  # Extra wait for stability
    
    # Check GPU memory
    vram_before = check_gpu_memory()
    
    # Send inference request
    prompt = "Hello " * (tokens // 2)  # Approximate token count
    
    curl_cmd = [
        "curl", "-s", "http://localhost:8005/v1/completions",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0
        })
    ]
    
    try:
        result = subprocess.run(curl_cmd, capture_output=True, timeout=60)
        request_success = result.returncode == 0
    except subprocess.TimeoutExpired:
        request_success = False
        print("Request timeout")
    
    time.sleep(2)
    
    # Measure VRAM
    vram_after = check_gpu_memory()
    
    # Get disk cache size
    try:
        result = subprocess.run(
            ["ls", "-la", cache_dir],
            capture_output=True, text=True
        )
        disk_size = 0
        for line in result.stdout.split('\n'):
            if '.pt' in line:
                size = int(line.split()[4])
                disk_size += size
        disk_size_mb = disk_size / (1024 * 1024)
    except:
        disk_size_mb = 0
    
    # Read LMCache VRAM log
    compression_vram = 0
    try:
        with open("/tmp/lmcache_vram.log", "r") as f:
            log_content = f.read()
            # Look for encode increase
            for line in log_content.split('\n'):
                if 'encode_function:' in line and 'increase=' in line:
                    try:
                        import re
                        match = re.search(r'increase=([0-9.]+)GB', line)
                        if match:
                            compression_vram = float(match.group(1))
                    except:
                        pass
    except:
        pass
    
    # Kill vLLM
    kill_vllm()
    
    result = {
        "status": "success",
        "mode": mode,
        "tokens": tokens,
        "gpu_memory_utilization": gpu_mem,
        "vram_before_gb": vram_before,
        "vram_after_gb": vram_after,
        "disk_size_mb": disk_size_mb,
        "compression_vram_gb": compression_vram,
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"Result: VRAM={vram_before:.2f}GB -> {vram_after:.2f}GB, Disk={disk_size_mb:.2f}MB")
    
    return result

def save_progress(results, filename):
    """Save progress to file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def load_progress(filename):
    """Load progress from file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        return []

def main():
    print("="*60)
    print("LMCache VRAM Sweep Experiment")
    print("="*60)
    
    # Check GPU first
    free_mem = check_gpu_memory()
    print(f"Free GPU memory: {free_mem:.2f} GB")
    
    if free_mem < 4:
        print("WARNING: Low GPU memory. Some tests may fail.")
    
    progress_file = "/tmp/sweep_progress.json"
    results = load_progress(progress_file)
    
    print(f"Previous results: {len(results)}")
    
    # Run CacheGen experiments
    print("\n" + "="*60)
    print("PHASE 1: CacheGen Mode")
    print("="*60)
    
    config_file_cachegen = "/home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen.yaml"
    
    for tokens, gpu_mem in SAFE_CONFIGS:
        key = f"cachegen_{tokens}_{gpu_mem}"
        
        # Check if already done
        if any(r.get("key") == key for r in results):
            print(f"Skipping {key} (already done)")
            continue
        
        try:
            result = run_experiment(tokens, gpu_mem, "cachegen", config_file_cachegen)
            result["key"] = key
            results.append(result)
            save_progress(results, progress_file)
            
            if result["status"] == "failed" and "oom" in str(result.get("error", "")).lower():
                print(f"OOM detected! Reducing test range...")
                break
                
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "key": key,
                "status": "error",
                "error": str(e)
            })
            save_progress(results, progress_file)
    
    # Run Native experiments  
    print("\n" + "="*60)
    print("PHASE 2: Native Mode")
    print("="*60)
    
    config_file_native = "/home/noslab-gpu/tkdgjs/experiment/lmcache_native.yaml"
    
    for tokens, gpu_mem in SAFE_CONFIGS:
        key = f"native_{tokens}_{gpu_mem}"
        
        # Check if already done
        if any(r.get("key") == key for r in results):
            print(f"Skipping {key} (already done)")
            continue
        
        try:
            result = run_experiment(tokens, gpu_mem, "native", config_file_native)
            result["key"] = key
            results.append(result)
            save_progress(results, progress_file)
            
            if result["status"] == "failed" and "oom" in str(result.get("error", "")).lower():
                print(f"OOM detected! Reducing test range...")
                break
                
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "key": key,
                "status": "error",
                "error": str(e)
            })
            save_progress(results, progress_file)
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    
    # Save final results
    output_file = "/tmp/sweep_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print(f"Total experiments: {len(results)}")
    
    # Print summary
    cachegen_results = [r for r in results if r.get("mode") == "cachegen" and r["status"] == "success"]
    native_results = [r for r in results if r.get("mode") == "native" and r["status"] == "success"]
    
    print(f"\nCacheGen: {len(cachegen_results)} successful")
    print(f"Native: {len(native_results)} successful")
    
    return results

if __name__ == "__main__":
    main()
