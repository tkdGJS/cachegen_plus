#!/usr/bin/env python3
"""
Sweep VRAM Experiment - Test multiple configurations
- GPU Memory Utilization: 0.5, 0.7, 0.9
- Prefill sizes: 256, 512, 1024, 2048, 4096
- Modes: native, cachegen
"""
import os
import sys
import time
import json
import subprocess
import signal
import requests
import threading
import re
import shutil
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

VLLM_PORT = 8000
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
EXPERIMENT_DIR = "/home/noslab-gpu/tkdgjs/experiment"

# LMCache config files - MUST use experiment directory versions
LMCACHE_CACHEGEN_CONFIG = "/home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen.yaml"
LMCACHE_NATIVE_CONFIG = "/home/noslab-gpu/tkdgjs/experiment/lmcache_native.yaml"

VLLM_BIN = "/home/noslab-gpu/tkdgjs/tkdgjs/bin/vllm"

# Sweep parameters
GPU_MEMORY_UTILIZATIONS = [0.5, 0.7, 0.9]
PREFILL_SIZES = [256, 512, 1024, 2048, 4096]
MODES = ["native", "cachegen"]


@dataclass
class VRAMSnapshot:
    """VRAM State Snapshot - 5 Regions (sum = nvidia-smi used)"""
    total_vram_gb: float = 0.0
    used_vram_gb: float = 0.0
    free_vram_gb: float = 0.0
    
    model_weights_gb: float = 0.0
    vllm_kv_cache_allocated_gb: float = 0.0
    vllm_kv_blocks_total: int = 0
    vllm_kv_blocks_free: int = 0
    vllm_kv_cache_used_gb: float = 0.0
    vllm_kv_blocks_used: int = 0
    vllm_kv_usage_ratio: float = 0.0
    activation_tensors_gb: float = 0.0
    cuda_runtime_gb: float = 0.0
    torch_allocated_gb: float = 0.0
    torch_reserved_gb: float = 0.0
    torch_peak_gb: float = 0.0
    estimated_cachegen_buffer_gb: float = 0.0
    sum_validated_gb: float = 0.0
    sum_diff_gb: float = 0.0


class FullVRAMMonitor:
    """Monitor 5 VRAM regions (sum = nvidia-smi used)"""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.baseline: Optional[VRAMSnapshot] = None
        
    def _get_nvidia_smi_memory(self) -> Dict[str, float]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free,memory.reserved", 
                 "--format=csv,noheader,nounits", "-i", "0"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                values = [float(x.strip()) for x in result.stdout.strip().split(',')]
                return {"total": values[0] / 1024, "used": values[1] / 1024, "free": values[2] / 1024, "reserved": values[3] / 1024}
        except:
            pass
        return {"total": 0, "used": 0, "free": 0, "reserved": 0}
    
    def _get_torch_memory(self) -> Dict:
        try:
            import torch
            if not torch.cuda.is_available():
                return {"allocated_gb": 0, "reserved_gb": 0, "peak_gb": 0}
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            return {"allocated_gb": allocated, "reserved_gb": reserved, "peak_gb": peak}
        except:
            return {"allocated_gb": 0, "reserved_gb": 0, "peak_gb": 0}
    
    def _get_vllm_kv_stats(self) -> Dict:
        try:
            resp = requests.get(f"http://localhost:{self.port}/metrics", timeout=5)
            if resp.status_code != 200:
                return {"blocks_total": 0, "blocks_free": 0, "blocks_used": 0, "usage_ratio": 0.0, "allocated_gb": 0.0, "used_gb": 0.0}
            
            text = resp.text
            blocks_total = 0
            block_size = 16
            
            match = re.search(r'num_gpu_blocks="(\d+)"', text)
            if match:
                blocks_total = int(match.group(1))
            
            match = re.search(r'block_size="(\d+)"', text)
            if match:
                block_size = int(match.group(1))
            
            usage_ratio = 0.0
            match = re.search(r'vllm:kv_cache_usage_perc\{[^}]*\} (\d+\.?\d*)', text)
            if match:
                usage_ratio = float(match.group(1))
            
            # Calculate actual KV memory: blocks_used * block_size
            # blocks_used = blocks_total * usage_ratio (if usage_ratio available)
            # Otherwise: estimate based on nvidia-smi VRAM
            
            # This is the MAXIMUM possible KV cache (token budget), not actual usage
            max_kv_gb = blocks_total * block_size / 1024
            
            # Actual KV usage: max_kv_gb * usage_ratio (or just max_kv if ratio unavailable but GPU is utilizing)
            # But we cap it to reasonable VRAM (nvidia-smi used VRAM - model weights - overhead)
            if usage_ratio > 0:
                actual_kv_gb = max_kv_gb * usage_ratio
            else:
                # If no usage ratio, assume minimal KV usage for idle state
                actual_kv_gb = 0.0
            
            # Cap to reasonable values (can't exceed total VRAM)
            total_vram_match = re.search(r'gpu_memory_utilization="(\d+\.?\d*)"', text)
            if total_vram_match:
                total_kv_budget_gb = 15.0 * float(total_vram_match.group(1))  # 15GB GPU * util ratio
                actual_kv_gb = min(actual_kv_gb, total_kv_budget_gb)
            
            return {"blocks_total": blocks_total, "block_size": block_size, "usage_ratio": usage_ratio, "max_kv_gb": max_kv_gb, "actual_kv_gb": actual_kv_gb}
        except:
            return {"blocks_total": 0, "blocks_free": 0, "blocks_used": 0, "usage_ratio": 0.0, "allocated_gb": 0.0, "used_gb": 0.0}
    
    def measure(self, baseline: Optional[VRAMSnapshot] = None) -> VRAMSnapshot:
        nvidia = self._get_nvidia_smi_memory()
        used_vram = nvidia.get("used", 0)
        vllm_kv = self._get_vllm_kv_stats()
        kv_max_gb = vllm_kv.get("max_kv_gb", 0)
        kv_actual_gb = vllm_kv.get("actual_kv_gb", 0)
        torch_mem = self._get_torch_memory()
        
        model_weights_gb = 0.0
        kv_used_gb = 0.0
        activation_gb = 0.0
        cuda_runtime_gb = 0.0
        
        # Calculate VRAM regions: nvidia-smi used = model + kv_actual + activation + cuda
        # kv_max_gb is token budget (may not be fully allocated)
        # kv_actual_gb is actual VRAM usage based on usage_ratio
        
        if baseline:
            model_weights_gb = baseline.used_vram_gb - baseline.vllm_kv_cache_allocated_gb
            if model_weights_gb < 0:
                model_weights_gb = baseline.used_vram_gb
        else:
            if kv_actual_gb > 0 and kv_actual_gb <= used_vram:
                model_weights_gb = max(0, used_vram - kv_actual_gb)
            else:
                model_weights_gb = used_vram * 0.80
                kv_used_gb = used_vram * 0.10
                cuda_runtime_gb = used_vram * 0.10
        
        if baseline:
            activation_gb = max(0, torch_mem["peak_gb"] - baseline.torch_allocated_gb)
        else:
            activation_gb = torch_mem.get("peak_gb", 0)
        
        if baseline or (kv_actual_gb > 0 and kv_actual_gb <= used_vram):
            kv_used_gb = kv_actual_gb if kv_actual_gb > 0 else kv_max_gb * vllm_kv.get("usage_ratio", 0)
            cuda_runtime_gb = max(0, used_vram - model_weights_gb - kv_used_gb - activation_gb)
        
        sum_regions = model_weights_gb + kv_used_gb + activation_gb + cuda_runtime_gb
        sum_diff = used_vram - sum_regions
        
        return VRAMSnapshot(
            total_vram_gb=nvidia.get("total", 0),
            used_vram_gb=used_vram,
            free_vram_gb=nvidia.get("free", 0),
            model_weights_gb=model_weights_gb,
            vllm_kv_cache_allocated_gb=kv_max_gb,
            vllm_kv_blocks_total=vllm_kv.get("blocks_total", 0),
            vllm_kv_blocks_free=0,
            vllm_kv_cache_used_gb=kv_used_gb,
            vllm_kv_blocks_used=0,
            vllm_kv_usage_ratio=vllm_kv.get("usage_ratio", 0),
            activation_tensors_gb=activation_gb,
            cuda_runtime_gb=cuda_runtime_gb,
            torch_allocated_gb=torch_mem.get("allocated_gb", 0),
            torch_reserved_gb=torch_mem.get("reserved_gb", 0),
            torch_peak_gb=torch_mem.get("peak_gb", 0),
            sum_validated_gb=sum_regions,
            sum_diff_gb=sum_diff,
        )
    
    def set_baseline(self) -> VRAMSnapshot:
        self.baseline = self.measure()
        return self.baseline


def get_disk_usage(path: str) -> Tuple[int, int]:
    if not os.path.exists(path):
        return 0, 0
    total_size = 0
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
                file_count += 1
    return file_count, total_size


def clear_lmcache_disk(serde_type: str):
    disk_path = f"{EXPERIMENT_DIR}/lmcache_{serde_type}_disk"
    if os.path.exists(disk_path):
        for f in os.listdir(disk_path):
            fp = os.path.join(disk_path, f)
            if os.path.isfile(fp):
                os.remove(fp)
            elif os.path.isfile(fp):
                shutil.rmtree(fp)
    else:
        os.makedirs(disk_path, exist_ok=True)
    return disk_path


def wait_for_vllm(port, timeout=180) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
            if resp.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False


def create_lmcache_config(serde_type: str, output_path: str) -> str:
    disk_path = f"{EXPERIMENT_DIR}/lmcache_{serde_type}_disk"
    os.makedirs(disk_path, exist_ok=True)
    
    config = f"""chunk_size: 256
save_unfull_chunk: true
local_cpu: true
max_local_cpu_size: 4.0
local_disk: "file://{disk_path}/"
max_local_disk_size: 10.0
enable_async_loading: true
enable_kv_events: true
remote_serde: {serde_type}
internal_api_server_enabled: true
internal_api_server_port_start: 6999
enable_chunk_statistics: true
chunk_statistics_strategy: "memory_bloom_filter"
chunk_statistics_auto_start_statistics: true
"""
    with open(output_path, 'w') as f:
        f.write(config)
    return disk_path


def start_vllm(serde_type: str, gpu_mem_util: float) -> Optional[subprocess.Popen]:
    if serde_type == "cachegen":
        config_path = LMCACHE_CACHEGEN_CONFIG
        disk_path = f"{EXPERIMENT_DIR}/lmcache_cachegen_disk"
    else:
        config_path = LMCACHE_NATIVE_CONFIG
        disk_path = f"{EXPERIMENT_DIR}/lmcache_torch_disk"
    
    os.makedirs(disk_path, exist_ok=True)
    
    env = os.environ.copy()
    env["LMCACHE_CONFIG_FILE"] = config_path
    env["PYTHONHASHSEED"] = "0"
    env["VLLM_DEBUG_MFU_METRICS"] = "1"
    
    kv_config = '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"discard_partial_chunks":false}}'
    
    subprocess.run(["pkill", "-f", "vllm serve"], stderr=subprocess.DEVNULL)
    time.sleep(3)
    
    cmd = [
        VLLM_BIN, "serve", MODEL,
        "--port", str(VLLM_PORT),
        "--dtype", "half",
        "--max-model-len", "8192",
        "--max-num-seqs", "128",
        "--max-num-batched-tokens", "4096",
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--disable-hybrid-kv-cache-manager",
        "--kv-transfer-config", kv_config,
        "--scheduling-policy", "fcfs",
        "--enable-chunked-prefill",
        "--enforce-eager",
        "--attention-backend", "triton_attn",
    ]
    
    print(f"[vLLM] Starting with {serde_type} mode, gpu_mem_util={gpu_mem_util}...")
    
    log_file = f"{EXPERIMENT_DIR}/vllm_{serde_type}_gm{gpu_mem_util}.log"
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=f)
    
    if wait_for_vllm(VLLM_PORT, timeout=180):
        print(f"[vLLM] Started successfully")
        return proc
    else:
        print(f"[vLLM] Failed to start!")
        return None


def stop_vllm():
    subprocess.run(["pkill", "-f", "vllm serve"], stderr=subprocess.DEVNULL)
    time.sleep(3)
    print("[vLLM] Stopped")


def send_request_and_measure(prompt_tokens: int, max_tokens: int = 32) -> Tuple[bool, dict]:
    prompt = "word " * prompt_tokens
    url = f"http://localhost:{VLLM_PORT}/v1/completions"
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}
    
    print(f"[Request] Sending {prompt_tokens} tokens...")
    
    start_time = time.time()
    first_token_time = None
    token_times = []
    
    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                print(f"[Request] Error: {resp.status_code}")
                return False, {"error": resp.text}
            
            for line in resp.iter_lines():
                if line:
                    current_time = time.time()
                    if first_token_time is None:
                        first_token_time = current_time
                    token_times.append(current_time)
            
            end_time = time.time()
    except Exception as e:
        print(f"[Request] Exception: {e}")
        return False, {"error": str(e)}
    
    ttft = (first_token_time - start_time) if first_token_time else 0.0
    total_time = end_time - start_time
    
    return True, {
        "success": True,
        "prompt_tokens": prompt_tokens,
        "output_tokens": len(token_times),
        "ttft_sec": round(ttft, 4),
        "ttlt_sec": round(total_time, 4)
    }


def run_single_experiment(serde_type: str, prefill_size: int, gpu_mem_util: float) -> dict:
    print(f"\n{'='*60}")
    print(f"Experiment: mode={serde_type}, prefill={prefill_size}, gpu_mem={gpu_mem_util}")
    print(f"{'='*60}")
    
    clear_lmcache_disk(serde_type)
    
    vllm_proc = start_vllm(serde_type, gpu_mem_util)
    if not vllm_proc:
        return {"error": "Failed to start vLLM"}
    
    warmup_sec = 30
    print(f"[Warmup] Waiting {warmup_sec}s for vLLM to stabilize...")
    time.sleep(warmup_sec)
    
    monitor = FullVRAMMonitorLoop(port=VLLM_PORT)
    disk_path = f"{EXPERIMENT_DIR}/lmcache_{serde_type}_disk"
    before_files, before_size = get_disk_usage(disk_path)
    
    vram_log_file = f"{EXPERIMENT_DIR}/vram_timeseries_{serde_type}_p{prefill_size}_gm{gpu_mem_util}.jsonl"
    
    monitor.start()
    time.sleep(2)
    
    request_start_time = time.time()
    success, latency_data = send_request_and_measure(prefill_size)
    
    time.sleep(10)
    
    monitor.stop()
    snapshot = monitor.get_snapshot()
    after_files, after_size = get_disk_usage(disk_path)
    
    samples = monitor.get_samples()
    
    with open(vram_log_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"[VRAM Log] Saved {len(samples)} samples to {vram_log_file}")
    
    stop_vllm()
    
    result = {
        "mode": serde_type,
        "prefill_size": prefill_size,
        "gpu_memory_utilization": gpu_mem_util,
        "success": success,
        "latency": latency_data,
        "vram": {
            "idle_used_vram_gb": snapshot.used_vram_gb,
            "model_weights_gb": snapshot.model_weights_gb,
            "vllm_kv_cache_allocated_gb": snapshot.vllm_kv_cache_allocated_gb,
            "vllm_kv_cache_used_gb": snapshot.vllm_kv_cache_used_gb,
            "vllm_kv_blocks_total": snapshot.vllm_kv_blocks_total,
            "vllm_kv_blocks_used": snapshot.vllm_kv_blocks_used,
            "activation_tensors_gb": snapshot.activation_tensors_gb,
            "cuda_runtime_gb": snapshot.cuda_runtime_gb,
            "sum_validated_gb": snapshot.sum_validated_gb,
            "sum_diff_gb": snapshot.sum_diff_gb,
        },
        "disk": {
            "offloaded_size_mb": round((after_size - before_size) / (1024**2), 2)
        }
    }
    
    print(f"[Result] used_vram={snapshot.used_vram_gb:.2f}GB, "
          f"kv_allocated={snapshot.vllm_kv_cache_allocated_gb:.2f}GB, "
          f"kv_used={snapshot.vllm_kv_cache_used_gb:.2f}GB, "
          f"sum_diff={snapshot.sum_diff_gb:.4f}GB")
    
    return result


class FullVRAMMonitorLoop:
    def __init__(self, interval: float = 0.1, port: int = 8000):
        self.interval = interval
        self.monitor = FullVRAMMonitor(port)
        self.running = False
        self.samples = []
        self.start_time = 0.0
        self.baseline: Optional[VRAMSnapshot] = None
        self._lock = threading.Lock()
        self.current_snapshot = None
        
    def _monitor_loop(self):
        while self.running:
            snapshot = self.monitor.measure(self.baseline)
            elapsed = time.time() - self.start_time
            timestamp = time.time()
            
            sample = asdict(snapshot)
            sample['elapsed_sec'] = elapsed
            sample['timestamp'] = timestamp
            
            with self._lock:
                self.samples.append(sample)
                self.current_snapshot = snapshot
            
            time.sleep(self.interval)
    
    def start(self):
        self.running = True
        self.start_time = time.time()
        self.samples = []
        self.baseline = self.monitor.set_baseline()
        
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
    
    def get_snapshot(self) -> VRAMSnapshot:
        with self._lock:
            return self.current_snapshot if self.current_snapshot else self.baseline
    
    def get_samples(self) -> List[Dict]:
        with self._lock:
            return self.samples.copy()


def main():
    print("="*60)
    print("SWEEP VRAM EXPERIMENT")
    print("="*60)
    print(f"Model: {MODEL}")
    print(f"GPU Memory Utilizations: {GPU_MEMORY_UTILIZATIONS}")
    print(f"Prefill sizes: {PREFILL_SIZES}")
    print(f"Modes: {MODES}")
    print("="*60)
    
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    
    all_results = {}
    
    for gpu_mem_util in GPU_MEMORY_UTILIZATIONS:
        all_results[f"gm_{gpu_mem_util}"] = {}
        
        for mode in MODES:
            all_results[f"gm_{gpu_mem_util}"][mode] = {}
            
            for prefill_size in PREFILL_SIZES:
                try:
                    result = run_single_experiment(mode, prefill_size, gpu_mem_util)
                    all_results[f"gm_{gpu_mem_util}"][mode][prefill_size] = result
                except Exception as e:
                    print(f"[ERROR] {mode}, {prefill_size}, {gpu_mem_util}: {e}")
                    all_results[f"gm_{gpu_mem_util}"][mode][prefill_size] = {"error": str(e)}
                
                time.sleep(30)
    
    final_file = f"{EXPERIMENT_DIR}/sweep_results.json"
    with open(final_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SWEEP EXPERIMENT COMPLETE")
    print(f"Results saved to: {final_file}")
    print("="*60)
    
    # Print summary
    print("\n=== SUMMARY ===")
    for gm_key, gm_data in all_results.items():
        print(f"\n{gm_key}:")
        for mode, mode_data in gm_data.items():
            print(f"  {mode}:")
            for pref, data in mode_data.items():
                if isinstance(data, dict) and "vram" in data:
                    vram = data["vram"]
                    print(f"    {pref} tokens: used_vram={vram.get('idle_used_vram_gb', 'N/A'):.2f}GB, "
                          f"kv_alloc={vram.get('vllm_kv_cache_allocated_gb', 'N/A'):.2f}GB, "
                          f"kv_used={vram.get('vllm_kv_cache_used_gb', 'N/A'):.2f}GB")


if __name__ == "__main__":
    main()
