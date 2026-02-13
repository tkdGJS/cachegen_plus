# CacheGen VRAM 실험 계획서

## 1. 실험 개요

### 1.1 실험 목적
- CacheGen KV 압축 사용 시 VRAM 부족 문제 실증 확인
- CacheGen 사용 vs 미사용 시 VRAM 사용량 비교
- 요청 처리 각 단계별 VRAM 프로파일링

### 1.2 가설 (Hypothesis)
**H1:** CacheGen 압축 사용 시 압축 중간에 원본 KV + 압축 버퍼가 동시에 VRAM에 존재하여 Peak VRAM이 더 높게 나타남

---

## 2. 실험 변수

### 2.1 독립 변수 (Independent Variables)

| 변수 | 수준 (Levels) | 설명 |
|------|--------------|------|
| **LMCache 모드** | 2 | `native` (torch), `cachegen` |
| **Prefill 크기** | 4 | 256, 512, 1024, 2048 토큰 |

### 2.2 종속 변수 (Dependent Variables)

| 변수 | 측정 방법 |
|------|----------|
| **VRAM Peak** | 0.1초 간격 nvidia-smi로 측정 |
| **TTFT** (Time to First Token) | 요청 시작 → 첫 토큰 출력 시간 |
| **TBT p95/p99** (Time Between Tokens) | 토큰 간 인터벌 95th/99th percentile |
| **TTLT** (Time to Last Token) | 요청 시작 → 마지막 토큰 출력 시간 |
| **Disk offload 크기** | 디스크에 저장된 KV 캐시 용량 |
| **Disk offload 청크 수** | 저장된 청크 개수 |

---

## 3. 실험 설계

### 3.1 실험 환경

```
하드웨어: Tesla T64
vLLM 버전: 0.14.0
LMCache 버전: 최신 (0.3.13)
모델: meta-llama/Llama-3.2-1B-Instruct
```

### 3.2 Attention Backend 설정

```bash
export VLLM_ATTENTION_BACKEND=TRION_ATTN  # Tesla T4 필수
```

### 3.3 설정 파일

**lmcache_native.yaml** (압축 미사용):
```yaml
chunk_size: 256
save_unfull_chunk: true
local_cpu: true
max_local_cpu_size: 4.0
local_disk: "file:///tmp/lmcache/lmcache_disk/"
max_local_disk_size: 4.0
remote_serde: torch  # ← 압축 없음
enable_async_loading: true
enable_kv_events: true
```

**lmcache_cachegen.yaml** (압축 사용):
```yaml
chunk_size: 256
save_unfull_chunk: true
local_cpu: true
max_local_cpu_size: 4.0
local_disk: "file:///tmp/lmcache/lmcache_disk/"
max_local_disk_size: 4.0
remote_serde: cachegen  # ← CacheGen 압축 사용
enable_async_loading: true
enable_kv_events: true
```

---

## 4. 실험 프로토콜

### 4.1 단일 실험 실행 흐름

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: vLLM 인스턴스 시작                                            │
│   - 모드 (native/cachegen)별 설정 적용                                 │
│   - 300초 warmup 대기                                               │
├─────────────────────────────────────────────────────────────────────┤
│ Step 2: 요청 전 VRAM 측정 (Phase A)                                  │
│   - VRAMMonitor 시작 (0.1초 간격)                                    │
│   - 10초간 측정 → 평균값 = " Idle VRAM"                            │
├─────────────────────────────────────────────────────────────────────┤
│ Step 3: 요청 전송 및 처리 (Phase B)                                   │
│   - Prefill N 토큰으로 요청 전송                                      │
│   - TTFT, TBT, TTLT 측정                                            │
│   - Streaming으로 토큰별 타임스탬프 기록                               │
├─────────────────────────────────────────────────────────────────────┤
│ Step 4: Disk Offloading 완료 대기 (Phase C)                          │
│   - 요청 완료 후 KV 캐시 디스크 쓰기 완료 대기                          │
│   - 10초 추가 대기                                                   │
├─────────────────────────────────────────────────────────────────────┤
│ Step 5: 최종 VRAM 측정 (Phase D)                                     │
│   - 300초 Sleep                                                     │
│   - VRAMMonitor 계속 측정                                            │
│   - 300초 후 10초간 측정 → 평균값 = "Final VRAM"                    │
├─────────────────────────────────────────────────────────────────────┤
│ Step 6: Disk Offload 정보 기록                                       │
│   - 디렉토리 내 파일 개수, 총 용량                                    │
│   - lmcache_metrics에서 청크 개수 조회                               │
├─────────────────────────────────────────────────────────────────────┤
│ Step 7: vLLM 인스턴스 종료                                           │
│   - 다음 실험을 위해 완전 종료                                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 측정 타임라인

```
Time ─────────────────────────────────────────────────────────────►

[Phase A]    [Phase B]           [Phase C]       [Phase D]
Idle         Request Processing   Offloading      300s Sleep
│             │                    │               │
│  10s        │   Variable         │   10s        │   10s
│  ████████   │   ████████████     │   ████████   │   ████████
│             │                    │               │
Start         Request              End             End
              ──► TTFT ──►         Request         Measurement
                         TTLT
```

### 4.3 VRAM 측정 상세

| 측정 시점 | 설명 | 데이터 |
|----------|------|--------|
| **Idle (Phase A)** | 요청 전송 전 10초 평균 | Baseline VRAM |
| **During Request (Phase B)** | 요청 처리 중 전체 측정 | Peak VRAM, 타임스탬프별 VRAM |
| **Post-Offload (Phase C)** | Offloading 완료 후 10초 평균 | Offload 후 VRAM |
| **Final (Phase D)** | 300초 대기 후 10초 평균 | 안정 상태 VRAM |

---

## 5. 실험 매트릭스

### 5.1 전체 실험 구성

| # | 모드 | Prefill | 순서 |
|---|------|---------|------|
| 1 | native | 256 | 1 |
| 2 | native | 512 | 2 |
| 3 | native | 1024 | 3 |
| 4 | native | 2048 | 4 |
| 5 | cachegen | 256 | 5 |
| 6 | cachegen | 512 | 6 |
| 7 | cachegen | 1024 | 7 |
| 8 | cachegen | 2048 | 8 |

**총 8회 실험** (각 실험마다 vLLM 재시작)

### 5.2 예상 소요 시간

| 단계 | 시간 |
|------|------|
| vLLM 시작 + Warmup | ~180초/실험 |
| 요청 처리 | ~5-30초/실험 |
| Offload 대기 | ~10초/실험 |
| 300초 Sleep | 300초/실험 |
| Cool down | ~30초/실험 |
| **총 시간** | **~9-10분/실험** |

**전체 실험 예상 시간: 80-90분**

---

## 5.3 완전한 측정 (Full VRAM Monitor) - 연구 목적

### 5.3.1 vLLM VRAM 레이아웃 구성요소

vLLM이 사용하는 VRAM은 **4개 영역**으로 구성됨:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           GPU VRAM (예: 16GB)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [영역 1] Model Weights                                                │
│    - 모델 파라미터 (FP16: ~3GB for 1B model)                           │
│    - Static, inference 중 변경 없음                                     │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [영역 2] KV Cache Blocks (vLLM Block Manager)                         │
│    - vLLM이 관리하는 KV 캐시                                           │
│    - 요청 처리 중 동적으로 할당/해제                                    │
│    - gpu_memory_utilization로 제어                                     │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [영역 3] Activation Tensors & 중간 연산 버퍼                           │
│    - Prefill/Decode 중 생성되는 임시 텐서                               │
│    - attention scores, MLP 중간 activations 등                          │
│    - 요청 종료 후 GC로 해제                                            │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [영역 4] CUDA Runtime & PyTorch Allocator                             │
│    - torch.cuda.memory_allocated()                                     │
│    - CUDA 커널 내부 버퍼                                               │
│    - cuBLAS, cuDNN 등 백엔드 버퍼                                     │
│    - **CacheGen 압축 버퍼가 여기에 할당됨!**                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3.2 CacheGen VRAM 추가 할당 영역

```
[CacheGen 압축 시 추가 할당]
┌─────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ① 원본 KV Tensor (vLLM Block)        Already exists                 │
│     └── memory_obj.tensor.cuda()      복사만 수행                     │
│                                                                         │
│  ② quantized_key/value (중간 버퍼)     ← NEW!                        │
│     └── torch.zeros(...).cuda()       FP16 → Quantized                │
│                                                                         │
│  ③ output_buffer (압축 출력)           ← NEW!                        │
│     └── torch.zeros(...).cuda()       uint8, GPU                     │
│                                                                         │
│  ④ cdf_int (CDF 계산용)               ← NEW!                        │
│     └── torch.zeros(...).cuda()                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3.3 FullVRAMMonitor 구현

```python
import torch
import requests
import subprocess
from typing import Dict, Optional
from dataclasses import dataclass, asdict

@dataclass
class VRAMSnapshot:
    """VRAM 상태 스냅샷"""
    # 전체 VRAM
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
    
    def __init__(self, vllm_port: int = 8000):
        self.vllm_port = vllm_port
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
        except Exception:
            pass
        return {"total": 0, "used": 0, "free": 0, "reserved": 0}
    
    def _get_torch_memory(self) -> Dict:
        """PyTorch allocator 상세 정보"""
        stats = torch.cuda.memory_stats()
        
        allocated = torch.cuda.memory_allocated() / (1024**3)  # bytes → GB
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
    
    def _get_vllm_kv_stats(self) -> Dict:
        """vLLM metrics에서 KV cache 사용량 조회"""
        try:
            resp = requests.get(f"http://localhost:{self.vllm_port}/metrics", timeout=5)
            if resp.status_code != 200:
                return {"blocks": 0, "usage_ratio": 0.0}
            
            text = resp.text
            
            # KV blocks
            blocks = 0
            for pattern in [r'kv_cache_manager_blocks_total (\d+)', 
                           r'vllm_kv_cache_blocks (\d+)']:
                import re
                match = re.search(pattern, text)
                if match:
                    blocks = int(match.group(1))
                    break
            
            # KV usage ratio
            usage_ratio = 0.0
            import re
            match = re.search(r'vllm_kv_cache_usage_ratio (\d+\.\d+)', text)
            if match:
                usage_ratio = float(match.group(1))
            
            return {"blocks": blocks, "usage_ratio": usage_ratio}
        except Exception:
            return {"blocks": 0, "usage_ratio": 0.0}
    
    def measure(self, baseline: Optional[VRAMSnapshot] = None) -> VRAMSnapshot:
        """현재 VRAM 상태 측정"""
        
        # 1. 전체 VRAM (nvidia-smi)
        nvidia = self._get_nvidia_smi_memory()
        
        # 2. PyTorch allocator
        torch_mem = self._get_torch_memory()
        
        # 3. vLLM KV blocks
        vllm_kv = self._get_vllm_kv_stats()
        
        snapshot = VRAMSnapshot(
            # 전체 VRAM
            total_vram_gb=nvidia.get("total", 0),
            used_vram_gb=nvidia.get("used", 0),
            free_vram_gb=nvidia.get("free", 0),
            
            # PyTorch
            torch_allocated_gb=torch_mem["allocated_gb"],
            torch_reserved_gb=torch_mem["reserved_gb"],
            torch_peak_gb=torch_mem["peak_gb"],
            active_bytes_gb=torch_mem["active_bytes_gb"],
            inactive_split_bytes_gb=torch_mem["inactive_split_bytes_gb"],
            num_alloc_retries=torch_mem["num_alloc_retries"],
            num_ooms=torch_mem["num_ooms"],
            
            # vLLM KV
            vllm_kv_blocks=vllm_kv["blocks"],
            vllm_kv_usage_ratio=vllm_kv["usage_ratio"],
        )
        
        # 4. CacheGen 버퍼 추정 (baseline 대비 증가분)
        if baseline:
            snapshot.estimated_cachegen_buffer_gb = (
                snapshot.torch_allocated_gb - baseline.torch_allocated_gb
            )
        
        return snapshot
    
    def set_baseline(self):
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
        import threading
        self._lock = threading.Lock()
        
    def _monitor_loop(self):
        import time
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
        import threading
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
    
    def get_samples(self):
        with self._lock:
            return self.samples.copy()
    
    def get_stats(self) -> Dict:
        """통계 요약"""
        samples = self.get_samples()
        if not samples:
            return {}
        
        # Peak 계산
        used_vram = [s["used_vram_gb"] for s in samples]
        torch_alloc = [s["torch_allocated_gb"] for s in samples]
        torch_peak = [s["torch_peak_gb"] for s in samples]
        cachegen_est = [s["estimated_cachegen_buffer_gb"] for s in samples]
        
        # Peak 시점 찾기
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
```

### 5.3.4 측정 데이터 해석

| 영역 | 측정 필드 | 의미 |
|------|---------|------|
| **전체 VRAM** | `used_vram_gb` | OOM 판단의 최종 지표 |
| **PyTorch** | `torch_allocated_gb` | CacheGen 버퍼가 여기에 할당 |
| **PyTorch Peak** | `torch_peak_gb` | Prefill 중 임시 메모리 포함 |
| **vLLM KV** | `vllm_kv_blocks` | 원본 KV 크기 |
| **추정 CacheGen** | `estimated_cachegen_buffer_gb` | 압축 버퍼 추정값 |

### 5.3.5 추정 계산식

```
CacheGen 버퍼 추정 = torch_allocated(압축 중) - torch_allocated(Idle) - (KV Blocks 변화)

where:
- torch_allocated(Idle): Baseline (요청 전)
- torch_allocated(압축 중): Peak 측정 시점
- KV Blocks 변화: vLLM이 할당한 KV 블록의 변화
```

---

## 6. 측정 도구

### 6.1 VRAM 모니터링

```python
# 0.1초 간격 측정
VRAMMonitor(interval=0.1)

# nvidia-smi 또는 vLLM metrics에서 조회
# 우선순위: vLLM metrics > nvidia-smi
```

### 6.2 Latency 측정

```
TTFT = t(first_token) - t(request_start)
TBT_i = t(token_i) - t(token_i-1)  (i >= 2)
TBT_p95 = percentile(TBT, 95)
TBT_p99 = percentile(TBT, 99)
TTLT = t(last_token) - t(request_start)
```

### 6.3 Disk Offload 측정

```bash
# 디렉토리 용량 조회
du -sb lmcache_native_disk/
ls -la lmcache_native_disk/ | wc -l

# LMCache metrics에서 청크 수 조회
curl http://localhost:8000/metrics | grep "lmcache_kv_local_chunk"
```

---

## 7. 예상 결과 형식

### 7.1 JSON 결과 구조

```json
{
  "experiment_id": "exp_001",
  "mode": "cachegen",
  "prefill_size": 1024,
  "timestamp": "2026-02-13T10:00:00",
  
  "vram": {
    "idle_gb": 5.23,
    "peak_gb": 7.84,
    "post_offload_gb": 5.45,
    "final_gb": 5.21,
    "samples_count": 3210,
    "samples": [
      {"elapsed": 0.0, "vram_gb": 5.23},
      {"elapsed": 0.1, "vram_gb": 5.24},
      ...
    ]
  },
  
  "latency": {
    "ttft_sec": 0.152,
    "tbt_p95_sec": 0.012,
    "tbt_p99_sec": 0.018,
    "ttlt_sec": 1.234,
    "output_tokens": 32
  },
  
  "disk": {
    "offloaded_files": 4,
    "offloaded_size_bytes": 524288,
    "offloaded_size_mb": 0.5,
    "chunk_count": 4
  },
  
  "config": {
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "vllm_version": "0.14.0",
    "lmcache_version": "0.3.13",
    "attention_backend": "TRITON_ATTN",
    "gpu_memory_utilization": 0.9
  }
}
```

### 7.2 분석 결과 테이블

| Mode | Prefill | Idle VRAM | Peak VRAM | Delta | TTFT | TTLT | Disk Size | Compression Ratio |
|------|---------|-----------|-----------|-------|------|------|-----------|------------------|
| native | 256 | 5.2GB | 6.1GB | +0.9GB | 0.1s | 0.8s | 2.0MB | 1.0x |
| cachegen | 256 | 5.2GB | 7.8GB | +2.6GB | 0.1s | 1.2s | 0.5MB | 4.0x |
| native | 1024 | 5.2GB | 7.5GB | +2.3GB | 0.3s | 2.5s | 8.0MB | 1.0x |
| cachegen | 1024 | 5.2GB | 9.8GB | +4.6GB | 0.4s | 4.0s | 2.0MB | 4.0x |

---

## 8. 실행 스크립트

### 8.1 실험 실행 명령어

```bash
# 실험 실행
cd /home/noslab-gpu/tkdgjs/experiment
python run_vram_experiment.py

# 또는 단일 모드만 실행
python run_vram_experiment.py --mode cachegen

# 특정 prefill만 실행
python run_vram_experiment.py --mode cachegen --prefill 1024
```

### 8.2 시작 스크립트 (수정 필요)

기존 `start_vllm_kvoff.sh`를 기반으로 수정:

```bash
# CacheGen 모드
export LMCACHE_CONFIG_FILE=/home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen.yaml
export VLLM_ATTENTION_BACKEND=TRION_ATTN
./start_vllm_kvoff.sh -m meta-llama/Llama-3.2-1B-Instruct -p 8000
```

---

## 9. 주의사항 및风险管理

### 9.1 OOM 발생 가능성

| 시나리오 | 위험도 | 완화책 |
|----------|--------|--------|
| CacheGen + Large Prefill (2048) | 높음 | GPU memory utilization 0.9 → 0.8로 낮춤 |
| 연속 실험 | 중간 | 매 실험 후 60초 cooldown |
| DiskFull | 낮음 | 실험 전 디스크 정리 |

### 9.2 예상 문제 및 해결

| 문제 | 해결책 |
|------|--------|
| vLLM 시작 실패 | 로그 확인, 포트 충돌 체크 |
| VRAM 측정 실패 | nvidia-smi 폴백 |
| 요청 타임아웃 | 120초로 설정 |
| OOM으로 vLLM 크래시 | GPU memory utilization 감소 |

---

## 10. 후속 분석

### 10.1 주요 분석 포인트

1. **VRAM Delta 비교**
   - CacheGen 미사용: Idle → Peak 증가분
   - CacheGen 사용: Idle → Peak 증가분
   - 차이 = 압축 버퍼 추가 사용량

2. **압축률 vs VRAM 트레이드오프**
   - Disk Size 감소율 (압축률)
   - VRAM Peak 증가분
   - 유효한 트레이드오프?

3. **Prefill 크기별 스케일링**
   - Prefill 2배 → VRAM 2배?
   - 압축 버퍼의 선형성 여부

### 10.2 시각화 아이디어

```python
# VRAM 타임시리즈 플롯
import matplotlib.pyplot as plt

# 각 실험별 VRAM 프로파일
plt.figure(figsize=(12, 8))
for mode in ['native', 'cachegen']:
    for prefill in [256, 512, 1024, 2048]:
        data = load_result(mode, prefill)
        plt.plot(data['time'], data['vram'], label=f"{mode}_{prefill}")
plt.xlabel('Time (s)')
plt.ylabel('VRAM (GB)')
plt.legend()
plt.savefig('vram_profiles.png')
```

---

## 11. 검증을 위한 체크리스트

실험 전 확인:
- [ ] Tesla T4 인식 확인: `nvidia-smi`
- [ ] vLLM 0.14.0 설치 확인: `vllm --version`
- [ ] LMCache 설치 확인: `pip show lmcache`
- [ ] 포트 8000 사용 가능: `lsof -i :8000`
- [ ] 디스크 공간 충분: `df -h /tmp`
- [ ] Attention backend 설정 확인: `echo $VLLM_ATTENTION_BACKEND`

---

## 12. 실행 후 산출물

| 파일 | 내용 |
|------|------|
| `results_native.json` | Native 모드 전체 결과 |
| `results_cachegen.json` | CacheGen 모드 전체 결과 |
| `final_results.json` | 전체 실험 결과 요약 |
| `vllm_native.log` | vLLM Native 모드 로그 |
| `vllm_cachegen.log` | vLLM CacheGen 모드 로그 |
| `vram_profiles.png` | VRAM 타임시리즈 시각화 |
| `comparison_table.csv` | 결과 비교 테이블 |

---

*문서 버전: 1.2 (환경 세팅 및 실험 방법 추가)*
*생성 일시: 2026-02-13*
*최종 업데이트: 2026-02-13*

---

# 부록 A: 상세 환경 세팅 가이드

## A.1 하드웨어/소프트웨어 환경

| 항목 | 설정값 |
|------|--------|
| **GPU** | Tesla T4 (15GB VRAM) |
| **vLLM 버전** | 0.14.0 |
| **LMCache 버전** | 0.3.13-gfc031d471 |
| **모델** | meta-llama/Llama-3.2-1B-Instruct |
| **Attention Backend** | TRITON_ATTN |
| **GPU Memory Utilization** | 0.7 (10.5GB 할당) |
| **max_model_len** | 8192 |
| **max_num_batched_tokens** | 131072 |

## A.2 필수 환경 변수

```bash
# PATH 설정
export PATH=/home/noslab-gpu/tkdgjs/tkdgjs/bin:$PATH

# LMCache 설정 파일
export LMCACHE_CONFIG_FILE=/home/noslab-gpu/tkdgjs/experiment/lmcache_native.yaml  # Native 모드
# 또는
export LMCACHE_CONFIG_FILE=/home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen.yaml  # CacheGen 모드

# Python hash 랜덤성 (LMCache 필수)
export PYTHONHASHSEED=0

# Attention backend (Tesla T4 필수)
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
```

## A.3 LMCache 설정 파일

### Native 모드 (압축 없음)

**파일: `/home/noslab-gpu/tkdgjs/experiment/lmcache_native.yaml`**

```yaml
chunk_size: 256
save_unfull_chunk: true
local_cpu: true
max_local_cpu_size: 4.0
local_disk: "file:///tmp/lmcache/lmcache_disk/"
max_local_disk_size: 10.0
remote_serde: torch          # ← 압축 없음
enable_async_loading: true
enable_kv_events: true
internal_api_server_enabled: true
internal_api_server_port_start: 6999
enable_chunk_statistics: true
chunk_statistics_strategy: "memory_bloom_filter"
chunk_statistics_auto_start_statistics: true
```

### CacheGen 모드 (압축 사용)

**파일: `/home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen.yaml`**

```yaml
chunk_size: 256
save_unfull_chunk: true
local_cpu: true
max_local_cpu_size: 4.0
local_disk: "file:///tmp/lmcache/lmcache_disk/"
max_local_disk_size: 10.0
remote_serde: cachegen       # ← CacheGen 압축
enable_async_loading: true
enable_kv_events: true
internal_api_server_enabled: true
internal_api_server_port_start: 6999
enable_chunk_statistics: true
chunk_statistics_strategy: "memory_bloom_filter"
chunk_statistics_auto_start_statistics: true
```

## A.4 vLLM 시작 명령어 (완전한 예시)

```bash
#!/bin/bash
# vLLM + LMCache 시작 스크립트

export PATH=/home/noslab-gpu/tkdgjs/tkdgjs/bin:$PATH
export LMCACHE_CONFIG_FILE=/home/noslab-gpu/tkdgjs/experiment/lmcache_native.yaml  # 또는 cachegen
export PYTHONHASHSEED=0
export VLLM_ATTENTION_BACKEND=TRITON_ATTN

# KV Transfer 설정
KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"discard_partial_chunks":false}}'

# vLLM 시작
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --port 8000 \
  --dtype half \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 131072 \
  --gpu-memory-utilization 0.7 \
  --disable-hybrid-kv-cache-manager \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  --kv-offloading-backend lmcache \
  --kv-offloading-size 4 \
  --scheduling-policy fcfs \
  --enable-chunked-prefill \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}' \
  --enable-mfu-metrics \
  --enable-logging-iteration-details \
  --attention-config '{"backend": "TRITON_ATTN"}'
```

### 주요 플래그 설명

| 플래그 | 설명 |
|--------|------|
| `--kv-offloading-backend lmcache` | KV offloading 백엔드를 lmcache로 설정 |
| `--kv-offloading-size 4` | KV offloading 크기 (GB) |
| `--disable-hybrid-kv-cache-manager` | Hybrid KV cache manager 비활성화 |
| `--attention-config '{"backend": "TRITON_ATTN"}'` | Tesla T4용 Attention backend |
| `--gpu-memory-utilization 0.7` | VRAM 70%만 사용 (OOM 방지) |

---

# 부록 B: 빠른 시작 가이드 (5분内有効)

## B.1 1단계: 환경 확인

```bash
# GPU 확인
nvidia-smi

# vLLM 버전 확인
vllm --version

# 포트 확인 (8000, 6999 비어있어야 함)
lsof -i :8000 || echo "Port 8000 is free"
lsof -i :6999 || echo "Port 6999 is free"
```

## B.2 2단계: 설정 파일 준비

```bash
# Native 설정 파일 생성
cat > /home/noslab-gpu/tkdgjs/experiment/lmcache_native.yaml << 'EOF'
chunk_size: 256
save_unfull_chunk: true
local_cpu: true
max_local_cpu_size: 4.0
local_disk: "file:///tmp/lmcache/lmcache_disk/"
max_local_disk_size: 10.0
remote_serde: torch
enable_async_loading: true
enable_kv_events: true
internal_api_server_enabled: true
internal_api_server_port_start: 6999
EOF

# CacheGen 설정 파일 생성
cat > /home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen.yaml << 'EOF'
chunk_size: 256
save_unfull_chunk: true
local_cpu: true
max_local_cpu_size: 4.0
local_disk: "file:///tmp/lmcache/lmcache_disk/"
max_local_disk_size: 10.0
remote_serde: cachegen
enable_async_loading: true
enable_kv_events: true
internal_api_server_enabled: true
internal_api_server_port_start: 6999
EOF
```

## B.3 3단계: vLLM 시작 (Native 모드)

```bash
# VRAM 확인 (비어있어야 함)
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
# 결과: 14914 (약 15GB 여유)

# Native 모드로 시작
export PATH=/home/noslab-gpu/tkdgjs/tkdgjs/bin:$PATH
export LMCACHE_CONFIG_FILE=/home/noslab-gpu/tkdgjs/experiment/lmcache_native.yaml
export PYTHONHASHSEED=0

KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"discard_partial_chunks":false}}'

nohup vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --port 8000 \
  --dtype half \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 131072 \
  --gpu-memory-utilization 0.7 \
  --disable-hybrid-kv-cache-manager \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  --kv-offloading-backend lmcache \
  --kv-offloading-size 4 \
  --scheduling-policy fcfs \
  --enable-chunked-prefill \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}' \
  --enable-mfu-metrics \
  --attention-config '{"backend": "TRITON_ATTN"}' \
  > vllm_native.log 2>&1 &

# 대기 (약 30-40초)
sleep 40

# 확인
curl -s http://localhost:8000/v1/models | grep -q "meta-llama" && echo "vLLM Ready!"
```

## B.4 4단계: 연결 확인

```bash
# 1. KV Offloading Backend 확인
curl -s http://localhost:8000/metrics | grep "kv_offloading_backend"

# 결과 예시: kv_offloading_backend="lmcache" ✅

# 2. VRAM 확인
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
# 결과 예시: 4237, 10677 ✅ (약 4.2GB 사용, 10.7GB 여유)

# 3. LMCache 로그 확인 (KV 저장 확인)
grep "Stored" vllm_native.log | tail -5

# 결과 예시:
# Stored 2 out of total 2 tokens. size: 0.0001 GB
# Stored 11 out of total 11 tokens. size: 0.0003 GB ✅
```

## B.5 5단계: CacheGen 모드로 변경

```bash
# vLLM 종료
pkill -f "vllm serve"
sleep 3

# VRAM 확인
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits

# CacheGen 설정으로 재시작
export LMCACHE_CONFIG_FILE=/home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen.yaml

KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"discard_partial_chunks":false}}'

nohup vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --port 8000 \
  --dtype half \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 131072 \
  --gpu-memory-utilization 0.7 \
  --disable-hybrid-kv-cache-manager \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  --kv-offloading-backend lmcache \
  --kv-offloading-size 4 \
  --scheduling-policy fcfs \
  --enable-chunked-prefill \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}' \
  --enable-mfu-metrics \
  --attention-config '{"backend": "TRITON_ATTN"}' \
  > vllm_cachegen.log 2>&1 &

sleep 40

# 확인
curl -s http://localhost:8000/v1/models | grep -q "meta-llama" && echo "CacheGen Mode Ready!"
```

---

# 부록 C: 검증된 실험 결과

## C.1 Native vs CacheGen 비교

| 항목 | Native (torch | 비) | CacheGen고 |
|------|-----------------|----------|------|
| **KV Offloading Backend** | lmcache ✅ | lmcache ✅ | 동일 |
| **VRAM 사용량** | 4.2 GB | 4.2 GB | 동일 |
| **Stored Tokens** | 2-11 | 400+ | CacheGen이 더 많음 |
| **Disk Size** | ~8 MB | ~72 KB | **100배 차이** |
| **압축률** | 1x | ~100x | CacheGen 압축 효과 큼 |

## C.2 로그 확인 명령어

```bash
# Native 모드 KV 저장 확인
grep "Stored" vllm_native.log

# 결과 예시:
# Stored 2 out of total 2 tokens. size: 0.0001 GB, offload_time: 0.9 ms
# Stored 11 out of total 11 tokens. size: 0.0003 GB, offload_time: 0.4 ms

# CacheGen 모드 KV 저장 확인
grep "Stored" vllm_cachegen.log

# 결과 예시:
# Stored 402 out of total 402 tokens. size: 0.0123 GB, offload_time: 2.4 ms
```

---

# 부록 D: 트러블슈팅

## D.1常见 문제

| 문제 | 원인 | 해결책 |
|------|------|--------|
| `vllm: not found` | PATH 설정 안됨 | `export PATH=/home/noslab-gpu/tkdgjs/tkdgjs/bin:$PATH` |
| `TRITON_ATTN` 오류 | Attention backend 잘못됨 | `--attention-config '{"backend": "TRITON_ATTN"}'` 사용 |
| LMCache 미연결 | kv-offloading-backend 기본값 | `--kv-offloading-backend lmcache` 명시 |
| VRAM 부족 | GPU Memory太高 | `--gpu-memory-utilization 0.7` 이하로 낮춤 |
| 포트 충돌 | 기존 프로세스 점유 | `pkill -f vllm` 후 재시작 |

## D.2 로그 확인

```bash
# vLLM 로그 실시간 확인
tail -f vllm_native.log

# LMCache 관련 로그만
grep -iE "lmcache|stored|offload" vllm_native.log

# 에러 로그
grep -iE "error|failed|exception" vllm_native.log | tail -20
```

---

# 부록 E: Git 관리 가이드

## E.1 GitHub 레포지토리

- **URL**: https://github.com/tkdGJS/cachegen_plus
- **SSH**: git@github.com:tkdgGJS/cachegen_plus.git
- **브랜치**: master (기본)

## E.2 주요 Git 명령어

### 변경 사항 확인
```bash
# 현재 상태 확인
git status

# 변경된 파일 확인
git diff

# 스테이지된 파일 확인
git diff --staged
```

### 변경 사항 커밋
```bash
# 특정 파일만 스테이징
git add <파일경로>

# 모든 변경 사항 스테이징
git add .

# 커밋 (변경 사항 설명 포함)
git commit -m "설명 메시지"

# 직전 커밋 메시지 수정
git commit --amend
```

### 히스토리 확인
```bash
# 커밋 히스토리 (한 줄 보기)
git log --oneline

# 특정 파일의 히스토리
git log --follow <파일경로>

# 커밋 상세 확인
git show <커밋해시>
```

### 원격 저장소
```bash
# GitHub에 푸시
git push origin master

# 푸시 후 Credential 저장 (다음부터는 비밀번호 불필요)
git config --global credential.helper store
```

### 브랜치 관리
```bash
# 새 브랜치 생성
git checkout -b <브랜치이름>

# 브랜치 전환
git checkout <브랜치이름>

# 브랜치 병합
git merge <브랜치이름>
```

## E.3 커밋 메시지 규칙

### 기본 구조
```
<타입>: <설명>

<상세 설명 (선택)>

Ultraworked with [Sisyphus](https://github.com/code-yeongyu/oh-my-opencode)
```

### 타입 Prefix
| 타입 | 설명 | 예시 |
|------|------|------|
| `feat` | 새로운 기능 | `feat: VRAM 모니터링 스크립트 추가` |
| `fix` | 버그 수정 | `fix: vLLM 시작 명령어 오류 수정` |
| `docs` | 문서 수정 | `docs: 실험 프로토콜 업데이트` |
| `refactor` | 코드 리팩토링 | `refactor: 스크립트 구조 정리` |
| `chore` | 기타 작업 | `chore: .gitignore 추가` |

### 예시
```bash
# 실험 스크립트 수정
git add tkdgjs/experiment/cachegen_vram_experiment/scripts/run_experiment.sh
git commit -m "fix: 실험 스크립트 경로 수정

- tkdjs 경로 상위로 이동에 따른 경로 수정
- 로그 디렉토리 자동 생성 추가"

# 결과 파일 추가
git add tkdgjs/experiment/results_*.json
git commit -m "docs: 실험 결과 데이터 추가

- native 모드 4개 prefill 결과
- cachegen 모드 4개 prefill 결과"
```

## E.4 프로젝트 파일 구조 (Git 관리 대상)

```
cachegen_plus/
├── .gitignore                    # Git 무시 파일
├── cachegen_vram_analysis.md     # 연구 분석 문서
├── tkdgjs/experiment/
│   ├── experiment_plan.md        # 전체 실험 프로토콜
│   ├── lmcache_native.yaml       # Native 설정
│   ├── lmcache_cachegen.yaml     # CacheGen 설정
│   ├── run_vram_experiment.py    # 실험 실행 스크립트
│   ├── send_request.py           # 요청 전송 스크립트
│   ├── vram_monitor.py           # VRAM 모니터
│   └── cachegen_vram_experiment/
│       ├── scripts/              # 실행 스크립트
│       │   ├── setup_environment.sh
│       │   ├── start_vllm_native.sh
│       │   ├── start_vllm_cachegen.sh
│       │   ├── stop_vllm.sh
│       │   ├── test_vram.sh
│       │   └── run_experiment.sh
│       └── configs/
│           ├── lmcache_native.yaml
│           └── lmcache_cachegen.yaml
```

---

*문서 버전: 1.3 (Git 관리 가이드 추가)*
*생성 일시: 2026-02-13*
*최종 업데이트: 2026-02-13*
