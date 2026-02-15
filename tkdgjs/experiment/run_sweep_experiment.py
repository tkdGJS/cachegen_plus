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
PREFILL_SIZES = [256, 512, 1024, 2048, 4096, 8192]
MODES = ["native", "cachegen"]


@dataclass
class VRAMSnapshot:
    """VRAM State Snapshot - All regions (sum = nvidia-smi used)"""
    total_vram_gb: float = 0.0
    used_vram_gb: float = 0.0
    free_vram_gb: float = 0.0
    reserved_vram_gb: float = 0.0
    
    model_weights_gb: float = 0.0
    
    vllm_kv_cache_allocated_gb: float = 0.0
    vllm_kv_blocks_total: int = 0
    vllm_kv_blocks_free: int = 0
    vllm_kv_blocks_used: int = 0
    vllm_kv_usage_ratio: float = 0.0
    vllm_kv_cache_used_gb: float = 0.0
    
    activation_tensors_gb: float = 0.0
    
    cuda_runtime_gb: float = 0.0
    
    cachegen_encoder_gb: float = 0.0
    cachegen_decoder_gb: float = 0.0
    cachegen_compressed_kv_gb: float = 0.0
    cachegen_total_gb: float = 0.0
    
    is_cachegen_mode: bool = False
    
    sum_validated_gb: float = 0.0
    sum_diff_gb: float = 0.0
    
    elapsed_sec: float = 0.0
    timestamp: float = 0.0
    
    def to_layout_dict(self) -> Dict:
        layout = {
            "Model Weights": self.model_weights_gb,
            "KV Cache (Allocated)": self.vllm_kv_cache_allocated_gb,
            "KV Cache (Used)": self.vllm_kv_cache_used_gb,
            "Activation Tensors": self.activation_tensors_gb,
            "CUDA Runtime": self.cuda_runtime_gb,
        }
        if self.is_cachegen_mode:
            layout["CacheGen Encoder"] = self.cachegen_encoder_gb
            layout["CacheGen Decoder"] = self.cachegen_decoder_gb
            layout["CacheGen Compressed KV"] = self.cachegen_compressed_kv_gb
        return layout
    
    def print_layout(self, title: str = "VRAM Layout"):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        print(f"Total VRAM: {self.total_vram_gb:.2f} GB")
        print(f"Used: {self.used_vram_gb:.2f} GB | Free: {self.free_vram_gb:.2f} GB | Reserved: {self.reserved_vram_gb:.2f} GB")
        print(f"{'-'*60}")
        
        layout = self.to_layout_dict()
        max_gb = max(self.used_vram_gb, 0.1)
        
        for name, gb in layout.items():
            if gb > 0.001:
                bar_len = int(gb / max_gb * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                pct = (gb / self.used_vram_gb * 100) if self.used_vram_gb > 0 else 0
                print(f"  {name:25s} {gb:6.2f} GB [{bar}] {pct:5.1f}%")
        
        print(f"{'-'*60}")
        print(f"  Sum (validated):        {self.sum_validated_gb:.2f} GB")
        print(f"  Diff (nvidia-smi-used): {self.sum_diff_gb:+.2f} GB")
        print(f"{'='*60}")


class FullVRAMMonitor:
    """Monitor VRAM regions (sum = nvidia-smi used)"""
    
    def __init__(self, port: int = 8000, is_cachegen: bool = False):
        self.port = port
        self.is_cachegen = is_cachegen
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
    
    def _get_vllm_kv_stats(self) -> Dict:
        try:
            resp = requests.get(f"http://localhost:{self.port}/metrics", timeout=5)
            if resp.status_code != 200:
                return {"blocks_total": 0, "usage_ratio": 0.0, "allocated_gb": 0.0, "used_gb": 0.0}
            
            text = resp.text
            
            # Get gpu_memory_utilization
            gpu_util = 0.7
            match = re.search(r'gpu_memory_utilization="(\d+\.?\d*)"', text)
            if match:
                gpu_util = float(match.group(1))
            
            # Get num_gpu_blocks
            blocks_total = 0
            match = re.search(r'num_gpu_blocks="(\d+)"', text)
            if match:
                blocks_total = int(match.group(1))
            
            # Get block_size
            block_size = 16
            match = re.search(r'block_size="(\d+)"', text)
            if match:
                block_size = int(match.group(1))
            
            # Get KV usage ratio
            usage_ratio = 0.0
            match = re.search(r'vllm:kv_cache_usage_perc\{[^}]*\} (\d+\.?\d*)', text)
            if match:
                usage_ratio = float(match.group(1))
            
            # Get free blocks
            blocks_free = 0
            match = re.search(r'num_gpu_blocks_free="(\d+)"', text)
            if match:
                blocks_free = int(match.group(1))
            
            # Calculate allocated KV memory: blocks × block_size
            allocated_kv_gb = blocks_total * block_size / (1024 * 1024)
            
            # Used blocks calculation
            blocks_used = blocks_total - blocks_free
            used_kv_gb = blocks_used * block_size / (1024 * 1024)
            
            # Calculate usage_ratio from blocks if not available from metrics
            if usage_ratio == 0 and blocks_total > 0:
                usage_ratio = (blocks_used / blocks_total) * 100
            
            return {
                "blocks_total": blocks_total,
                "blocks_free": blocks_free,
                "blocks_used": blocks_used,
                "block_size": block_size,
                "gpu_utilization": gpu_util,
                "usage_ratio": usage_ratio,
                "allocated_gb": allocated_kv_gb,
                "used_gb": used_kv_gb
            }
        except:
            return {"blocks_total": 0, "blocks_free": 0, "blocks_used": 0, "block_size": 16, "usage_ratio": 0.0, "allocated_gb": 0.0, "used_gb": 0.0}
    
    def measure(self, baseline: Optional[VRAMSnapshot] = None, is_cachegen: bool = False) -> VRAMSnapshot:
        nvidia = self._get_nvidia_smi_memory()
        used_vram = nvidia.get("used", 0)
        reserved_vram = nvidia.get("reserved", 0)
        total_vram = nvidia.get("total", 15.0)
        
        vllm_kv = self._get_vllm_kv_stats()
        
        kv_allocated_gb = vllm_kv.get("allocated_gb", 0)
        kv_used_gb = vllm_kv.get("used_gb", 0)
        kv_blocks_total = vllm_kv.get("blocks_total", 0)
        kv_blocks_free = vllm_kv.get("blocks_free", 0)
        kv_blocks_used = vllm_kv.get("blocks_used", 0)
        usage_ratio = vllm_kv.get("usage_ratio", 0)
        
        vllm_running = kv_blocks_total > 0
        
        if baseline:
            model_weights_gb = baseline.model_weights_gb
            cuda_runtime_gb = baseline.cuda_runtime_gb
            activation_gb = baseline.activation_tensors_gb
        elif vllm_running:
            estimated_weights = 2.0
            estimated_cuda = 0.5
            activation_gb = max(0, used_vram - kv_used_gb - estimated_weights - estimated_cuda)
            model_weights_gb = estimated_weights
            cuda_runtime_gb = estimated_cuda
        else:
            model_weights_gb = 0.0
            cuda_runtime_gb = 0.0
            activation_gb = 0.0
        
        cachegen_encoder_gb = 0.0
        cachegen_decoder_gb = 0.0
        cachegen_compressed_kv_gb = 0.0
        cachegen_total_gb = 0.0
        
        if is_cachegen and baseline:
            vram_increase = used_vram - baseline.used_vram_gb
            if vram_increase > 0.1:
                encoder_estimate = 0.08
                decoder_estimate = 0.06
                temp_buffer_estimate = min(vram_increase * 0.5, 0.3)
                cachegen_encoder_gb = encoder_estimate
                cachegen_decoder_gb = decoder_estimate
                cachegen_compressed_kv_gb = temp_buffer_estimate
                cachegen_total_gb = cachegen_encoder_gb + cachegen_decoder_gb + cachegen_compressed_kv_gb
                cuda_runtime_gb = max(0, cuda_runtime_gb - cachegen_total_gb)
        
        sum_regions = model_weights_gb + kv_used_gb + activation_gb + cuda_runtime_gb + cachegen_total_gb
        sum_diff = used_vram - sum_regions
        
        return VRAMSnapshot(
            total_vram_gb=total_vram,
            used_vram_gb=used_vram,
            free_vram_gb=nvidia.get("free", 0),
            reserved_vram_gb=reserved_vram,
            model_weights_gb=model_weights_gb,
            vllm_kv_cache_allocated_gb=kv_allocated_gb,
            vllm_kv_blocks_total=kv_blocks_total,
            vllm_kv_blocks_free=kv_blocks_free,
            vllm_kv_blocks_used=kv_blocks_used,
            vllm_kv_usage_ratio=usage_ratio,
            vllm_kv_cache_used_gb=kv_used_gb,
            activation_tensors_gb=activation_gb,
            cuda_runtime_gb=cuda_runtime_gb,
            cachegen_encoder_gb=cachegen_encoder_gb,
            cachegen_decoder_gb=cachegen_decoder_gb,
            cachegen_compressed_kv_gb=cachegen_compressed_kv_gb,
            cachegen_total_gb=cachegen_total_gb,
            is_cachegen_mode=is_cachegen,
            sum_validated_gb=sum_regions,
            sum_diff_gb=sum_diff,
        )
    
    def set_baseline(self) -> VRAMSnapshot:
        self.baseline = self.measure(is_cachegen=self.is_cachegen)
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


def clear_gpu_processes():
    """Kill all GPU processes for stable experiment"""
    subprocess.run(["pkill", "-9", "-f", "vllm"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", "python.*serve"], stderr=subprocess.DEVNULL)
    time.sleep(3)
    
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True, timeout=5
    )
    used = float(result.stdout.strip()) / 1024
    
    if used > 1.0:
        print(f"[WARNING] GPU still has {used:.2f}GB used after cleanup")
        subprocess.run(["nvidia-smi", "--gpu-reset", "-i", "0"], stderr=subprocess.DEVNULL)
        time.sleep(5)
    
    return used


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
    
    monitor = FullVRAMMonitorLoop(interval=0.01, port=VLLM_PORT)
    disk_path = f"{EXPERIMENT_DIR}/lmcache_{serde_type}_disk"
    before_files, before_size = get_disk_usage(disk_path)
    
    vram_log_file = f"{EXPERIMENT_DIR}/vram_timeseries_{serde_type}_p{prefill_size}_gm{gpu_mem_util}.jsonl"
    
    # Start monitoring 30 seconds BEFORE request
    print(f"[Monitor] Starting VRAM monitoring 30s before request...")
    monitor.start()
    time.sleep(30)  # Wait 30 seconds before request (capture baseline stability)
    
    # Send request (monitoring is active)
    print(f"[Request] Sending request at {time.time() - monitor.start_time:.2f}s...")
    request_start_time = time.time()
    success, latency_data = send_request_and_measure(prefill_size)
    request_end_time = time.time()
    print(f"[Request] Request completed in {request_end_time - request_start_time:.2f}s")
    
    # Continue monitoring for 30 seconds AFTER request completes
    print(f"[Monitor] Continuing monitoring for 30s after request...")
    time.sleep(30)
    
    monitor.stop()
    snapshot = monitor.get_snapshot()
    after_files, after_size = get_disk_usage(disk_path)
    
    samples = monitor.get_samples()
    
    with open(vram_log_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"[VRAM Log] Saved {len(samples)} samples to {vram_log_file}")
    
    stop_vllm()
    
    peak_vram = monitor.get_peak_vram()
    
    result = {
        "mode": serde_type,
        "prefill_size": prefill_size,
        "gpu_memory_utilization": gpu_mem_util,
        "success": success,
        "latency": latency_data,
        "vram": {
            "idle_used_vram_gb": snapshot.used_vram_gb,
            "peak_vram_gb": snapshot.used_vram_gb + peak_vram,
            "peak_vram_increase_gb": peak_vram,
            "model_weights_gb": snapshot.model_weights_gb,
            "vllm_kv_cache_allocated_gb": snapshot.vllm_kv_cache_allocated_gb,
            "vllm_kv_cache_used_gb": snapshot.vllm_kv_cache_used_gb,
            "vllm_kv_blocks_total": snapshot.vllm_kv_blocks_total,
            "vllm_kv_blocks_used": snapshot.vllm_kv_blocks_used,
            "activation_tensors_gb": snapshot.activation_tensors_gb,
            "cuda_runtime_gb": snapshot.cuda_runtime_gb,
            "cachegen_total_gb": snapshot.cachegen_total_gb,
            "sum_validated_gb": snapshot.sum_validated_gb,
            "sum_diff_gb": snapshot.sum_diff_gb,
        },
        "disk": {
            "offloaded_size_mb": round((after_size - before_size) / (1024**2), 2)
        }
    }
    
    print(f"[Result] used_vram={snapshot.used_vram_gb:.2f}GB, "
          f"peak_increase={peak_vram:.2f}GB, "
          f"kv_allocated={snapshot.vllm_kv_cache_allocated_gb:.2f}GB, "
          f"kv_used={snapshot.vllm_kv_cache_used_gb:.2f}GB, "
          f"sum_diff={snapshot.sum_diff_gb:.4f}GB")
    
    return result


class FullVRAMMonitorLoop:
    def __init__(self, interval: float = 0.1, port: int = 8000, is_cachegen: bool = False):
        self.interval = interval
        self.is_cachegen = is_cachegen
        self.monitor = FullVRAMMonitor(port, is_cachegen=is_cachegen)
        self.running = False
        self.samples = []
        self.start_time = 0.0
        self.baseline: Optional[VRAMSnapshot] = None
        self._lock = threading.Lock()
        self.current_snapshot = None
        self.peak_vram_gb = 0.0
        self.peak_baseline_vram_gb = 0.0
        
    def _monitor_loop(self):
        while self.running:
            snapshot = self.monitor.measure(self.baseline, is_cachegen=self.is_cachegen)
            elapsed = time.time() - self.start_time
            
            if self.baseline:
                vram_increase = snapshot.used_vram_gb - self.baseline.used_vram_gb
                if vram_increase > self.peak_vram_gb:
                    self.peak_vram_gb = vram_increase
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
        self.peak_vram_gb = 0.0
        self.peak_baseline_vram_gb = self.baseline.used_vram_gb if self.baseline else 0.0
        
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
    
    def get_snapshot(self) -> Optional[VRAMSnapshot]:
        with self._lock:
            return self.current_snapshot if self.current_snapshot else self.baseline
    
    def get_samples(self) -> List[Dict]:
        with self._lock:
            return self.samples.copy()
    
    def get_peak_vram(self) -> float:
        with self._lock:
            return self.peak_vram_gb


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
