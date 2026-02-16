#!/usr/bin/env python3
"""
VRAM Experiment Sweep - Full Timeline Tracking
- Each sweep: start vLLM, run test, stop vLLM
- Save timeline to JSON file
- Record disk offload size, compression ratio, write amount
"""
import sys
import os
import time
import json
import subprocess
import shutil
import glob

EXPERIMENT_DIR = "/home/noslab-gpu/tkdgjs/experiment"
VLLM_BIN = "/home/noslab-gpu/tkdgjs/tkdgjs/bin/vllm"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
PORT = 8000

GPU_UTIL_VALUES = [0.3, 0.5, 0.7, 0.9]
PREFILL_SIZES = [128, 256, 512, 1024, 2048]
MODES = ["native", "cachegen"]

OOM_LOG_FILE = f"{EXPERIMENT_DIR}/oom_events.log"
VRAM_LOG_FILE = f"{EXPERIMENT_DIR}/vram_timeline.log"
SUMMARY_FILE = f"{EXPERIMENT_DIR}/sweep_results_latest.json"

def is_experiment_done(mode: str, prefill_size: int, gpu_util: float) -> bool:
    timeline_file = f"{EXPERIMENT_DIR}/timeline_{mode}_p{prefill_size}_gm{gpu_util}.jsonl"
    if not os.path.exists(timeline_file):
        return False
    try:
        with open(timeline_file, "r") as f:
            for line in f:
                data = json.loads(line)
                if data.get("type") == "result":
                    return data.get("success", False)
    except:
        pass
    return False

def cleanup_gpu():
    subprocess.run(["pkill", "-9", "-f", "vllm"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", "EngineCore"], stderr=subprocess.DEVNULL)
    time.sleep(3)

def clear_disk_cache(mode: str):
    disk_path = f"{EXPERIMENT_DIR}/lmcache_{mode}_disk"
    if os.path.exists(disk_path):
        shutil.rmtree(disk_path)
    os.makedirs(disk_path, exist_ok=True)
    return disk_path

def get_disk_size(path: str) -> int:
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total

def get_vram_usage():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            mem_str = result.stdout.strip().split('\n')[0]
            return float(mem_str) / 1024
    except:
        pass
    return 0.0

def log_oom_event(mode, prefill_size, gpu_util, stage, log_file, timestamp):
    with open(OOM_LOG_FILE, "a") as f:
        f.write(json.dumps({
            "timestamp": timestamp,
            "mode": mode,
            "prefill_size": prefill_size,
            "gpu_util": gpu_util,
            "stage": stage,
            "vram_gb": get_vram_usage(),
            "log_file": log_file
        }) + "\n")

def log_vram_sample(timestamp, mode, prefill_size, gpu_util, stage, vram_gb):
    with open(VRAM_LOG_FILE, "a") as f:
        f.write(json.dumps({
            "timestamp": timestamp,
            "mode": mode,
            "prefill_size": prefill_size,
            "gpu_util": gpu_util,
            "stage": stage,
            "vram_gb": vram_gb
        }) + "\n")

def check_oom_in_log(log_file):
    if not os.path.exists(log_file):
        return False, "unknown"
    
    oom_stages = {
        "startup": ["CUDA out of memory", "OOM during initialization"],
        "request": ["CUDA out of memory", "out of memory", "OutOfMemoryError"],
        "compression": ["CUDA out of memory during compression", "OOM during compression"]
    }
    
    try:
        with open(log_file, "r") as f:
            content = f.read()
            for stage, patterns in oom_stages.items():
                for pattern in patterns:
                    if pattern.lower() in content.lower():
                        return True, stage
    except:
        pass
    
    return False, "none"

def start_vllm(mode: str, gpu_util: float):
    config_file = f"{EXPERIMENT_DIR}/lmcache_{mode}.yaml"
    log_file = f"{EXPERIMENT_DIR}/vllm_{mode}_gm{gpu_util}.log"
    
    env = os.environ.copy()
    env["LMCACHE_CONFIG_FILE"] = config_file
    env["PYTHONPATH"] = "/home/noslab-gpu/tkdgjs/tkdgjs/lib/python3.10/site-packages"
    
    cmd = [
        VLLM_BIN, "serve", MODEL,
        "--port", str(PORT),
        "--dtype", "half",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", str(gpu_util),
        "--attention-backend", "TRITON_ATTN",
        "--kv-transfer-config", '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}',
    ]
    
    proc = subprocess.Popen(cmd, env=env, stdout=open(log_file, "w"), stderr=subprocess.STDOUT)
    return proc, log_file

def wait_for_vllm(timeout=180):
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

def send_request(prefill_size: int):
    prompt = "Hello world, this is a test. " * (prefill_size // 5)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 10,
        "temperature": 0.0
    }
    
    start = time.time()
    try:
        r = subprocess.run([
            "curl", "-s", "-X", "POST",
            f"http://localhost:{PORT}/v1/completions",
            "-H", "Content-Type: application/json",
            "-d", json.dumps(payload)
        ], capture_output=True, timeout=60)
        latency = time.time() - start
        success = r.returncode == 0
        return success, latency
    except Exception as e:
        return False, 0

def run_timeline_test(mode: str, prefill_size: int, gpu_util: float):
    print(f"\n{'='*60}")
    print(f"Testing: mode={mode}, prefill={prefill_size}, gpu_util={gpu_util}")
    print(f"{'='*60}")
    
    timestamp_start = time.time()
    oom_detected = False
    oom_stage = "none"
    
    cleanup_gpu()
    time.sleep(2)
    
    disk_path = clear_disk_cache(mode)
    
    print("Starting vLLM...")
    vram_before_start = get_vram_usage()
    log_vram_sample(timestamp_start, mode, prefill_size, gpu_util, "before_start", vram_before_start)
    
    proc, log_file = start_vllm(mode, gpu_util)
    
    if not wait_for_vllm(timeout=120):
        print(f"ERROR: vLLM not ready - checking for OOM...")
        has_oom, stage = check_oom_in_log(log_file)
        if has_oom:
            oom_detected = True
            oom_stage = stage
            log_oom_event(mode, prefill_size, gpu_util, stage, log_file, time.time())
            print(f"OOM detected at stage: {stage}")
        proc.kill()
        time.sleep(2)
        return {
            "mode": mode,
            "prefill_size": prefill_size,
            "gpu_util": gpu_util,
            "success": False,
            "error": "vLLM not ready",
            "oom_detected": oom_detected,
            "oom_stage": oom_stage,
            "vram_before_start_gb": vram_before_start
        }
    
    print("vLLM ready")
    vram_after_start = get_vram_usage()
    log_vram_sample(time.time(), mode, prefill_size, gpu_util, "after_start", vram_after_start)
    
    initial_disk_size = get_disk_size(disk_path)
    
    print(f"Sending request with {prefill_size} tokens...")
    vram_before_request = get_vram_usage()
    log_vram_sample(time.time(), mode, prefill_size, gpu_util, "before_request", vram_before_request)
    
    success, latency = send_request(prefill_size)
    
    vram_during_request = get_vram_usage()
    log_vram_sample(time.time(), mode, prefill_size, gpu_util, "during_request", vram_during_request)
    
    if not success:
        has_oom, stage = check_oom_in_log(log_file)
        if has_oom:
            oom_detected = True
            oom_stage = stage
            log_oom_event(mode, prefill_size, gpu_util, stage, log_file, time.time())
            print(f"OOM detected during request at stage: {stage}")
    
    time.sleep(5)
    
    final_disk_size = get_disk_size(disk_path)
    offloaded_size = final_disk_size - initial_disk_size
    
    vram_after_compression = get_vram_usage()
    log_vram_sample(time.time(), mode, prefill_size, gpu_util, "after_compression", vram_after_compression)
    
    compression_ratio = 0.0
    try:
        with open(log_file, "r") as f:
            for line in f:
                if "size:" in line and "Stored" in line:
                    parts = line.split("size:")
                    if len(parts) > 1:
                        size_str = parts[1].split("GB")[0].strip()
                        compression_ratio = float(size_str)
    except:
        pass
    
    proc.kill()
    time.sleep(2)
    
    result = {
        "mode": mode,
        "prefill_size": prefill_size,
        "gpu_util": gpu_util,
        "success": success,
        "latency_sec": latency,
        "disk_offload_bytes": offloaded_size,
        "disk_offload_mb": offloaded_size / (1024**2),
        "compression_ratio": compression_ratio,
        "kv_cache_size_gb": compression_ratio,
        "compressed_size_gb": compression_ratio * 0.3 if compression_ratio > 0 else 0,
        "oom_detected": oom_detected,
        "oom_stage": oom_stage,
        "vram_before_start_gb": vram_before_start,
        "vram_after_start_gb": vram_after_start,
        "vram_before_request_gb": vram_before_request,
        "vram_during_request_gb": vram_during_request,
        "vram_after_compression_gb": vram_after_compression,
    }
    
    print(f"Result: success={success}, latency={latency:.2f}s, "
          f"offloaded={result['disk_offload_mb']:.2f}MB, "
          f"compression_ratio={compression_ratio:.4f}, "
          f"oom={oom_detected}({oom_stage})")
    
    return result


def run_sweep():
    print("="*70)
    print("VRAM Experiment Sweep - Full Timeline")
    print("="*70)
    print(f"GPU Util values: {GPU_UTIL_VALUES}")
    print(f"Prefill sizes: {PREFILL_SIZES}")
    print(f"Modes: {MODES}")
    print(f"Total experiments: {len(GPU_UTIL_VALUES) * len(PREFILL_SIZES) * len(MODES)}")
    print("="*70)
    
    print("\n[CLEANUP] Cleaning up previous output files...")
    cleanup_gpu()
    time.sleep(2)
    
    if os.path.exists(OOM_LOG_FILE):
        os.remove(OOM_LOG_FILE)
    if os.path.exists(VRAM_LOG_FILE):
        os.remove(VRAM_LOG_FILE)
    
    results = []
    
    for gpu_util in GPU_UTIL_VALUES:
        for prefill_size in PREFILL_SIZES:
            for mode in MODES:
                if is_experiment_done(mode, prefill_size, gpu_util):
                    print(f"\n[SKIP] mode={mode}, prefill={prefill_size}, gpu_util={gpu_util} (already completed)")
                    timeline_file = f"{EXPERIMENT_DIR}/timeline_{mode}_p{prefill_size}_gm{gpu_util}.jsonl"
                    try:
                        with open(timeline_file, "r") as f:
                            for line in f:
                                data = json.loads(line)
                                if data.get("type") == "result":
                                    results.append(data)
                                    break
                    except:
                        pass
                    continue
                
                result = run_timeline_test(mode, prefill_size, gpu_util)
                results.append(result)
                
                timeline_file = f"{EXPERIMENT_DIR}/timeline_{mode}_p{prefill_size}_gm{gpu_util}.jsonl"
                with open(timeline_file, "w") as f:
                    f.write(json.dumps({
                        "type": "metadata",
                        "mode": mode,
                        "prefill_size": prefill_size,
                        "gpu_util": gpu_util,
                        "timestamp": time.time()
                    }) + "\n")
                    f.write(json.dumps({
                        "type": "result",
                        **result
                    }) + "\n")
                
                print(f"Saved timeline to {timeline_file}")
                
                time.sleep(3)
    
    # Save summary
    summary_file = f"{EXPERIMENT_DIR}/sweep_results_{int(time.time())}.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    
    with open(SUMMARY_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("Sweep Complete!")
    print("="*70)
    
    # Print summary
    print(f"\nTotal experiments: {len(results)}")
    successful = [r for r in results if r.get("success", False)]
    print(f"Successful: {len(successful)}")
    
    # Compare modes
    native_results = [r for r in results if r["mode"] == "native" and r.get("success")]
    cachegen_results = [r for r in results if r["mode"] == "cachegen" and r.get("success")]
    
    if native_results and cachegen_results:
        print(f"\n{'Metric':<30} {'Native':>12} {'CacheGen':>12}")
        print("-"*60)
        
        avg_latency_nat = sum(r["latency_sec"] for r in native_results) / len(native_results)
        avg_latency_cge = sum(r["latency_sec"] for r in cachegen_results) / len(cachegen_results)
        print(f"{'Avg Latency (s)':<30} {avg_latency_nat:>12.2f} {avg_latency_cge:>12.2f}")
        
        avg_offload_nat = sum(r["disk_offload_mb"] for r in native_results) / len(native_results)
        avg_offload_cge = sum(r["disk_offload_mb"] for r in cachegen_results) / len(cachegen_results)
        print(f"{'Avg Disk Offload (MB)':<30} {avg_offload_nat:>12.2f} {avg_offload_cge:>12.2f}")
        
        avg_ratio_nat = sum(r["compression_ratio"] for r in native_results) / len(native_results)
        avg_ratio_cge = sum(r["compression_ratio"] for r in cachegen_results) / len(cachegen_results)
        print(f"{'Avg Compression Ratio':<30} {avg_ratio_nat:>12.4f} {avg_ratio_cge:>12.4f}")
    
    print(f"\nResults saved to: {summary_file}")
    print("="*70)


if __name__ == "__main__":
    run_sweep()
