#!/usr/bin/env python3
"""
Complete VRAM + Latency Experiment for CacheGen vs Native comparison
- CacheGen mode: remote_serde=cachegen (KV 캐시 압축)
- Native mode: remote_serde=torch (압축 없이 디스크 오프로딩)

Full VRAM Monitor: Measures 4 VRAM regions
- Region 1: Model Weights (static)
- Region 2: KV Cache Blocks (vLLM)
- Region 3: Activation Tensors
- Region 4: CUDA Runtime & PyTorch Allocator (includes CacheGen buffers)
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
PREFILL_SIZES = [256, 512, 1024, 2048]
MODES = ["native", "cachegen"]
EXPERIMENT_DIR = "/home/noslab-gpu/tkdgjs/experiment"
VLLM_BASE_CONFIG = "/home/noslab-gpu/tkdgjs/qlm/qlm/endpoints/lmcache.yaml"
VLLM_BIN = "/home/noslab-gpu/tkdgjs/tkdgjs/bin/vllm"


@dataclass
class VRAMSnapshot:
    """VRAM 상태 스냅샷 - 4개 영역 측정"""
    # 전체 VRAM (nvidia-smi)
    total_vram_gb: float = 0.0
    used_vram_gb: float = 0.0
    free_vram_gb: float = 0.0
    
    # PyTorch allocator (영역 3, 4)
    torch_allocated_gb: float = 0.0
    torch_reserved_gb: float = 0.0
    torch_peak_gb: float = 0.0
    
    # PyTorch detailed stats
    active_bytes_gb: float = 0.0
    inactive_split_bytes_gb: float = 0.0
    num_alloc_retries: int = 0
    num_ooms: int = 0
    
    # vLLM KV Blocks (영역 2)
    vllm_kv_blocks: int = 0
    vllm_kv_usage_ratio: float = 0.0
    
    # CacheGen 버퍼 추정 (계산값)
    estimated_cachegen_buffer_gb: float = 0.0


class FullVRAMMonitor:
    """4개 VRAM 영역을 모두 추적하는 모니터"""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.baseline: Optional[VRAMSnapshot] = None
        
    def _get_nvidia_smi_memory(self) -> Dict[str, float]:
        """nvidia-smi로 전체 VRAM 조회"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free,memory.reserved", 
                 "--format=csv,noheader,nounits", "-i", "0"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                values = [float(x.strip()) for x in result.stdout.strip().split(',')]
                return {
                    "total": values[0] / 1024,  # MB → GB
                    "used": values[1] / 1024,
                    "free": values[2] / 1024,
                    "reserved": values[3] / 1024,
                }
        except Exception as e:
            print(f"[FullVRAMMonitor] nvidia-smi error: {e}")
        return {"total": 0, "used": 0, "free": 0, "reserved": 0}
    
    def _get_torch_memory(self) -> Dict:
        """PyTorch allocator 상세 정보"""
        try:
            import torch
            if not torch.cuda.is_available():
                return {"allocated_gb": 0, "reserved_gb": 0, "peak_gb": 0,
                        "active_bytes_gb": 0, "inactive_split_bytes_gb": 0,
                        "num_alloc_retries": 0, "num_ooms": 0}
            
            stats = torch.cuda.memory_stats()
            
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "peak_gb": peak,
                "active_bytes_gb": stats.get("active_bytes.all.current", 0) / (1024**3),
                "inactive_split_bytes_gb": stats.get("inactive_split_bytes.all.current", 0) / (1024**3),
                "num_alloc_retries": stats.get("num_alloc_retries", 0),
                "num_ooms": stats.get("num_ooms", 0),
            }
        except Exception as e:
            print(f"[FullVRAMMonitor] torch memory error: {e}")
            return {"allocated_gb": 0, "reserved_gb": 0, "peak_gb": 0,
                    "active_bytes_gb": 0, "inactive_split_bytes_gb": 0,
                    "num_alloc_retries": 0, "num_ooms": 0}
    
    def _get_vllm_kv_stats(self) -> Dict:
        """vLLM metrics에서 KV cache 사용량 조회"""
        try:
            resp = requests.get(f"http://localhost:{self.port}/metrics", timeout=5)
            if resp.status_code != 200:
                return {"blocks": 0, "usage_ratio": 0.0}
            
            text = resp.text
            
            # KV blocks
            blocks = 0
            for pattern in [r'kv_cache_manager_blocks_total (\d+)', 
                           r'vllm_kv_cache_blocks (\d+)']:
                match = re.search(pattern, text)
                if match:
                    blocks = int(match.group(1))
                    break
            
            # KV usage ratio
            usage_ratio = 0.0
            match = re.search(r'vllm_kv_cache_usage_ratio (\d+\.\d+)', text)
            if match:
                usage_ratio = float(match.group(1))
            
            return {"blocks": blocks, "usage_ratio": usage_ratio}
        except Exception:
            return {"blocks": 0, "usage_ratio": 0.0}
    
    def measure(self, baseline: Optional[VRAMSnapshot] = None) -> VRAMSnapshot:
        """현재 VRAM 상태 측정"""
        
        # 1. 전체 VRAM
        nvidia = self._get_nvidia_smi_memory()
        
        # 2. PyTorch allocator
        torch_mem = self._get_torch_memory()
        
        # 3. vLLM KV blocks
        vllm_kv = self._get_vllm_kv_stats()
        
        snapshot = VRAMSnapshot(
            total_vram_gb=nvidia.get("total", 0),
            used_vram_gb=nvidia.get("used", 0),
            free_vram_gb=nvidia.get("free", 0),
            torch_allocated_gb=torch_mem["allocated_gb"],
            torch_reserved_gb=torch_mem["reserved_gb"],
            torch_peak_gb=torch_mem["peak_gb"],
            active_bytes_gb=torch_mem["active_bytes_gb"],
            inactive_split_bytes_gb=torch_mem["inactive_split_bytes_gb"],
            num_alloc_retries=torch_mem["num_alloc_retries"],
            num_ooms=torch_mem["num_ooms"],
            vllm_kv_blocks=vllm_kv["blocks"],
            vllm_kv_usage_ratio=vllm_kv["usage_ratio"],
        )
        
        # 4. CacheGen 버퍼 추정
        if baseline:
            snapshot.estimated_cachegen_buffer_gb = (
                snapshot.torch_allocated_gb - baseline.torch_allocated_gb
            )
        
        return snapshot
    
    def set_baseline(self) -> VRAMSnapshot:
        """Baseline 설정 (Idle 상태)"""
        self.baseline = self.measure()
        return self.baseline


class FullVRAMMonitorLoop:
    """연속 측정 루프"""
    
    def __init__(self, interval: float = 0.1, port: int = 8000):
        self.interval = interval
        self.monitor = FullVRAMMonitor(port)
        self.running = False
        self.samples = []
        self.start_time = 0.0
        self.baseline: Optional[VRAMSnapshot] = None
        self._lock = threading.Lock()
        
    def _monitor_loop(self):
        while self.running:
            snapshot = self.monitor.measure(self.baseline)
            elapsed = time.time() - self.start_time
            
            with self._lock:
                self.samples.append({
                    "elapsed_sec": round(elapsed, 3),
                    "timestamp": time.time(),
                    **asdict(snapshot)
                })
            
            time.sleep(self.interval)
    
    def start(self):
        self.running = True
        self.start_time = time.time()
        self.samples = []
        self.baseline = self.monitor.set_baseline()
        
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"[FullVRAMMonitor] Started - baseline: {self.baseline.used_vram_gb:.2f} GB")
        
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        print(f"[FullVRAMMonitor] Stopped - collected {len(self.samples)} samples")
    
    def get_samples(self):
        with self._lock:
            return self.samples.copy()
    
    def get_stats(self) -> Dict:
        """통계 요약"""
        samples = self.get_samples()
        if not samples:
            return {}
        
        used_vram = [s["used_vram_gb"] for s in samples]
        torch_alloc = [s["torch_allocated_gb"] for s in samples]
        torch_peak = [s["torch_peak_gb"] for s in samples]
        cachegen_est = [s["estimated_cachegen_buffer_gb"] for s in samples]
        
        peak_idx = used_vram.index(max(used_vram))
        
        return {
            "idle": {
                "used_vram_gb": self.baseline.used_vram_gb,
                "torch_allocated_gb": self.baseline.torch_allocated_gb,
                "vllm_kv_blocks": self.baseline.vllm_kv_blocks,
            },
            "peak": {
                "used_vram_gb": max(used_vram),
                "torch_allocated_gb": max(torch_alloc),
                "torch_peak_gb": max(torch_peak),
                "estimated_cachegen_buffer_gb": max(cachegen_est),
                "peak_time_sec": samples[peak_idx]["elapsed_sec"],
            },
            "final": {
                "used_vram_gb": samples[-1]["used_vram_gb"],
                "torch_allocated_gb": samples[-1]["torch_allocated_gb"],
            },
            "sample_count": len(samples),
        }


# Legacy VRAMMonitor for backward compatibility
class VRAMMonitor:
    
    def __init__(self, interval=0.1, port=8000):
        self.interval = interval
        self.port = port
        self.running = False
        self.samples = []
        self.start_time = 0.0
        self._lock = threading.Lock()
        
    def get_vram_from_nvidia_smi(self) -> Optional[float]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip()) / 1024
        except:
            pass
        return None
    
    def get_vram_from_vllm(self) -> Optional[float]:
        try:
            resp = requests.get(f"http://localhost:{self.port}/metrics", timeout=5)
            if resp.status_code != 200:
                return self.get_vram_from_nvidia_smi()
            
            match = re.search(r'vllm_gpu_memory_usage_bytes\{gpu="0"\} (\d+\.?\d*e?[+-]?\d*)', resp.text)
            if match:
                bytes_used = float(match.group(1))
                return bytes_used / (1024**3)
            
            match = re.search(r'vllm_gpu_memory_bytes\{type="used",gpu="0"\} (\d+)', resp.text)
            if match:
                bytes_used = float(match.group(1))
                return bytes_used / (1024**3)
                
            return self.get_vram_from_nvidia_smi()
        except Exception as e:
            return self.get_vram_from_nvidia_smi()
    
    def monitor_loop(self):
        while self.running:
            vram_gb = self.get_vram_from_vllm()
            if vram_gb is not None:
                elapsed = time.time() - self.start_time
                with self._lock:
                    self.samples.append({
                        "elapsed_sec": round(elapsed, 3),
                        "timestamp": time.time(),
                        "vram_gb": round(vram_gb, 4)
                    })
            time.sleep(self.interval)
    
    def start(self):
        self.running = True
        self.start_time = time.time()
        self.samples = []
        self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.thread.start()
        print(f"[VRAMMonitor] Started monitoring vLLM port {self.port}")
        
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        print(f"[VRAMMonitor] Stopped. Collected {len(self.samples)} samples")
        
    def get_samples(self) -> List[dict]:
        with self._lock:
            return self.samples.copy()
    
    def get_stats(self) -> dict:
        samples = self.get_samples()
        if not samples:
            return {}
        vram_values = [s["vram_gb"] for s in samples]
        return {
            "min_gb": min(vram_values),
            "max_gb": max(vram_values),
            "peak_gb": max(vram_values),
            "sample_count": len(vram_values)
        }


def get_disk_usage(path: str) -> Tuple[int, int]:
    """디렉토리 용량 조회 (files, size_bytes)"""
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
            elif os.path.isdir(fp):
                shutil.rmtree(fp)
        print(f"[Disk] Cleared {disk_path}")
    else:
        os.makedirs(disk_path, exist_ok=True)
        print(f"[Disk] Created {disk_path}")
    return disk_path


def wait_for_vllm(port, timeout=180) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
            if resp.status_code == 200:
                print(f"[vLLM] Ready on port {port}")
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
    print(f"[Config] Created {output_path} with remote_serde: {serde_type}, disk: {disk_path}")
    return disk_path


def start_vllm(serde_type: str) -> Optional[subprocess.Popen]:
    if serde_type == "cachegen":
        config_path = VLLM_BASE_CONFIG
        disk_path = f"{EXPERIMENT_DIR}/lmcache_cachegen_disk"
    else:
        config_path = f"{EXPERIMENT_DIR}/lmcache_native.yaml"
        disk_path = create_lmcache_config("torch", config_path)
    
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
        "--gpu-memory-utilization", "0.9",
        "--disable-hybrid-kv-cache-manager",
        "--kv-transfer-config", kv_config,
        "--scheduling-policy", "fcfs",
        "--enable-chunked-prefill",
        "--enforce-eager",
        "--attention-backend", "triton_attn",
    ]
    
    print(f"[vLLM] Starting with {serde_type} mode...")
    print(f"[vLLM] Command: {' '.join(cmd)}")
    print(f"[vLLM] LMCACHE_CONFIG_FILE: {config_path}")
    print(f"[vLLM] VLLM_ATTENTION_BACKEND: TRITON_ATTN")
    
    log_file = f"{EXPERIMENT_DIR}/vllm_{serde_type}.log"
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


def get_chunk_count_from_vllm() -> Optional[int]:
    """vLLM metrics에서 청크 개수 조회"""
    try:
        resp = requests.get(f"http://localhost:{VLLM_PORT}/metrics", timeout=5)
        if resp.status_code != 200:
            return None
        
        text = resp.text
        
        match = re.search(r'lmcache_kv_local_chunk_count_total (\d+)', text)
        if match:
            return int(match.group(1))
        
        match = re.search(r'vllm_cache_stat_total_chunks (\d+)', text)
        if match:
            return int(match.group(1))
        
        return None
    except Exception as e:
        print(f"Error getting chunk count: {e}", file=sys.stderr)
        return None


def send_request_and_measure(prompt_tokens: int, max_tokens: int = 32) -> Tuple[bool, dict]:
    
    prompt = "word " * prompt_tokens
    
    url = f"http://localhost:{VLLM_PORT}/v1/completions"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0
    }
    
    print(f"[Request] Sending {prompt_tokens} tokens (max_tokens={max_tokens})...")
    
    start_time = time.time()
    first_token_time = None
    token_times = []
    
    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                print(f"[Request] Error: {resp.status_code} - {resp.text}")
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
    
    total_time = end_time - start_time
    
    ttft = (first_token_time - start_time) if first_token_time else 0.0
    
    if len(token_times) > 1:
        inter_token_times = [token_times[i+1] - token_times[i] for i in range(len(token_times)-1)]
        inter_token_times.sort()
        
        n = len(inter_token_times)
        tbt_p95 = inter_token_times[int(n * 0.95)] if n > 0 else 0
        tbt_p99 = inter_token_times[int(n * 0.99)] if n > 0 else 0
    else:
        tbt_p95 = 0
        tbt_p99 = 0
    
    ttlt = total_time
    
    result = {
        "success": True,
        "prompt_tokens": prompt_tokens,
        "output_tokens": len(token_times),
        "ttft_sec": round(ttft, 4),
        "tbt_p95_sec": round(tbt_p95, 4),
        "tbt_p99_sec": round(tbt_p99, 4),
        "ttlt_sec": round(ttlt, 4)
    }
    
    print(f"[Request] Done - TTFT: {result['ttft_sec']}s, TBT_p95: {result['tbt_p95_sec']}s, TBT_p99: {result['tbt_p99_sec']}s, TTLT: {result['ttlt_sec']}s")
    
    return True, result


def run_single_experiment(serde_type: str, prefill_size: int) -> dict:
    """단일 실험 실행 (vLLM 시작/종료 포함)"""
    
    print(f"\n{'='*60}")
    print(f"Experiment: mode={serde_type}, prefill={prefill_size}")
    print(f"{'='*60}")
    
    clear_lmcache_disk(serde_type)
    
    vllm_proc = start_vllm(serde_type)
    if not vllm_proc:
        return {"error": "Failed to start vLLM"}
    
    warmup_sec = 30
    print(f"[Warmup] Waiting {warmup_sec}s for vLLM to stabilize...")
    time.sleep(warmup_sec)
    
    # Use FullVRAMMonitorLoop for 4-region VRAM measurement
    monitor = FullVRAMMonitorLoop(interval=0.1, port=VLLM_PORT)
    
    disk_path = f"{EXPERIMENT_DIR}/lmcache_{serde_type}_disk"
    before_files, before_size = get_disk_usage(disk_path)
    
    monitor.start()
    time.sleep(2)
    
    success, latency_data = send_request_and_measure(prefill_size)
    
    time.sleep(10)
    
    monitor.stop()
    vram_samples = monitor.get_samples()
    vram_stats = monitor.get_stats()
    
    after_files, after_size = get_disk_usage(disk_path)
    
    chunk_count = get_chunk_count_from_vllm()
    
    stop_vllm()
    
    # Build result with FullVRAMMonitor's detailed stats
    # Note: Don't save raw_samples to JSON (too large for repeated experiments)
    result = {
        "mode": serde_type,
        "prefill_size": prefill_size,
        "success": success,
        "latency": latency_data,
        "vram": vram_stats,  # FullVRAMMonitor provides idle/peak/final breakdown
        "disk": {
            "before_files": before_files,
            "before_size_bytes": before_size,
            "after_files": after_files,
            "after_size_bytes": after_size,
            "offloaded_files": after_files - before_files,
            "offloaded_size_bytes": after_size - before_size,
            "offloaded_size_mb": round((after_size - before_size) / (1024**2), 2)
        },
        "chunk_count": chunk_count,
        "vram_samples_count": len(vram_samples)
    }
    
    # Print detailed results for FullVRAMMonitor
    print(f"[Result] === Full VRAM Monitor Results ===")
    if "idle" in vram_stats:
        print(f"[Result] Idle:   used_vram={vram_stats['idle'].get('used_vram_gb', 'N/A')}GB, "
              f"torch_alloc={vram_stats['idle'].get('torch_allocated_gb', 'N/A')}GB, "
              f"kv_blocks={vram_stats['idle'].get('vllm_kv_blocks', 'N/A')}")
    if "peak" in vram_stats:
        print(f"[Result] Peak:   used_vram={vram_stats['peak'].get('used_vram_gb', 'N/A')}GB, "
              f"torch_alloc={vram_stats['peak'].get('torch_allocated_gb', 'N/A')}GB, "
              f"torch_peak={vram_stats['peak'].get('torch_peak_gb', 'N/A')}GB, "
              f"cachegen_est={vram_stats['peak'].get('estimated_cachegen_buffer_gb', 'N/A')}GB")
    if "final" in vram_stats:
        print(f"[Result] Final:  used_vram={vram_stats['final'].get('used_vram_gb', 'N/A')}GB, "
              f"torch_alloc={vram_stats['final'].get('torch_allocated_gb', 'N/A')}GB")
    print(f"[Result] Samples: {vram_stats.get('sample_count', len(vram_samples))}")
    print(f"[Result] Disk offloaded: {result['disk']['offloaded_files']} files, {result['disk']['offloaded_size_mb']} MB")
    print(f"[Result] Chunk count: {chunk_count}")
    
    return result


def run_experiment_for_mode(serde_type: str) -> dict:
    """한 모드에 대한 전체 실험"""
    
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT MODE: {serde_type}")
    print(f"#{'#'*60}")
    
    results = {}
    
    for prefill_size in PREFILL_SIZES:
        result = run_single_experiment(serde_type, prefill_size)
        results[prefill_size] = result
        
        time.sleep(10)
    
    results_file = f"{EXPERIMENT_DIR}/results_{serde_type}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[Results] Saved to {results_file}")
    
    print("[Cooldown] Waiting 60s before next mode...")
    time.sleep(60)
    
    return results


def main():
    print("="*60)
    print("CacheGen vs Native VRAM + Latency Experiment")
    print("="*60)
    print(f"Model: {MODEL}")
    print(f"Prefill sizes: {PREFILL_SIZES}")
    print(f"Modes: {MODES}")
    print(f"Attention Backend: TRITON_ATTN (Tesla T4)")
    print(f"vLLM: 0.14.0, LMCache: 0.3.13")
    print("="*60)
    
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    
    all_results = {}
    
    for mode in MODES:
        try:
            results = run_experiment_for_mode(mode)
            all_results[mode] = results
        except Exception as e:
            print(f"[ERROR] Mode {mode}: {e}")
            import traceback
            traceback.print_exc()
            all_results[mode] = {"error": str(e)}
    
    final_file = f"{EXPERIMENT_DIR}/final_results.json"
    with open(final_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {final_file}")
    
    print("\n=== SUMMARY (Full VRAM Monitor) ===")
    for mode in MODES:
        print(f"\n{mode.upper()}:")
        if mode in all_results and isinstance(all_results[mode], dict):
            for pref, data in all_results[mode].items():
                if pref.startswith("_"):
                    continue
                if isinstance(data, dict) and "latency" in data:
                    lat = data["latency"]
                    vram = data.get("vram", {})
                    disk = data.get("disk", {})
                    chunks = data.get("chunk_count")
                    
                    # Extract FullVRAMMonitor fields
                    idle = vram.get("idle", {})
                    peak = vram.get("peak", {})
                    
                    print(f"  {pref} tokens:")
                    print(f"    VRAM: idle={idle.get('used_vram_gb', 'N/A')}GB, "
                          f"peak={peak.get('used_vram_gb', 'N/A')}GB")
                    print(f"    PyTorch: idle={idle.get('torch_allocated_gb', 'N/A')}GB, "
                          f"peak={peak.get('torch_allocated_gb', 'N/A')}GB, "
                          f"torch_peak={peak.get('torch_peak_gb', 'N/A')}GB")
                    print(f"    KV Blocks: {idle.get('vllm_kv_blocks', 'N/A')}, "
                          f"CacheGen Buffer Est: {peak.get('estimated_cachegen_buffer_gb', 'N/A')}GB")
                    print(f"    Latency: TTFT={lat.get('ttft_sec', 'N/A')}s, "
                          f"TTLT={lat.get('ttlt_sec', 'N/A')}s")
                    print(f"    Disk: {disk.get('offloaded_size_mb', 'N/A')}MB, Chunks={chunks}")


if __name__ == "__main__":
    main()
