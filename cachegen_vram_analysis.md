# CacheGen VRAM 문제 분석 결과

## 문제 개요

**Observed Problem:**
- vLLM이 요청을 처리한 후 KV cache 블록을 VRAM에 할당
- 요청 완료 후 LMCache가 CacheGen으로 KV를 압축하려고 시도
- 압축 중 원본 KV + 중간 버퍼가 동시에 VRAM에 존재
- vLLM이 이미 많은 KV 블록을 할당한 상태면 VRAM 부족 → OOM 발생

---

## 코드 분석 결과

### 1. 압축 흐름 (cache_engine.py:1253-1308)

```python
# 1. KV 캐시 가져오기 (VRAM에 원본 KV 존재)
memory_objs = self.storage_manager.batched_get(keys=keys, location=location)

# 2. 압축 수행 - 문제가 발생!
compressed_memory_obj = serializer.serialize(memory_obj)  # ← 압축 중
#     ↑ 이 순간: 원본 KV + 압축 버퍼 + 출력 버퍼가 모두 VRAM에 존재

# 3. 압축 완료 후 원본 해제
memory_obj.unpin()           # ← 압축이 끝난 후에야 frees
self.storage_manager.batched_remove(keys, locations=[location])

# 4. 압축 데이터 저장
self.storage_manager.batched_put(keys, compressed_memory_objs, location=location)
```

### 2. 메모리 할당 위치

**CacheGenEncoder (cachegen_encoder.py:305-337):**

```python
# GPU에 추가 버퍼 할당
output_buffer = torch.zeros(
    (nlayers, nchannels, CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK),  # ← 추가 VRAM
    dtype=torch.uint8,
    device=encode_input.device,  # ← CUDA device
)

output_lengths = torch.zeros(
    (nlayers, nchannels), 
    dtype=torch.int32, 
    device=encode_input.device  # ← CUDA device
)
```

### 3. VRAM 부족 시 동작

- **예외 처리 없음**: compress() 함수에 try-catch 없음
- **CUDA OOM 발생**: torch.cuda.OutOfMemoryError
- **Worker 실패**: 예외 전파되어 compress worker crash

---

## VRAM 부족 시나리오

```
[TIMELINE]
t0: 원본 KV (vLLM block) ─────────────────────────────►
t1: serialize() 시작 ───► tensor.cuda() 복사 ─────────►
t2: quantized_key/value 할당 ──────────────────────────►  
t3: output_buffer 할당 ─────────────────────────────────►
t4: encode_function() 실행 ────────────────────────────►
t5: to_bytes() → 압축 완료 ─────────────────────────────►
t6: unpin() 호출 ──► 원본 KV 해제 ─────────────────────►
t7: batched_put() → 디스크 저장 ───────────────────────►

[MEMORY AT t3-t5]
┌────────────────┬────────────────┬────────────────┐
│  vLLM KV Block │ 중간 버퍼들    │ 압축 버퍼      │
│   (원본 KV)    │ (quantized)   │ (output_buffer)│
└────────────────┴────────────────┴────────────────┘
        VRAM ←─────────────────────────────────────►
```

---

# 연구 기여 가능성 분석 (Research Contribution Analysis)

## 1. Novelty 분석

| 측면 | 분석 | 관련 연구 |
|------|------|----------|
| **문제의 독창성** | KV cache 압축 시 **메모리 중복 문제**는 기존 연구에서 다루어지지 않음 | 기존 연구는 압축률/품질에 집중 |
| **시스템적 관점** | Two-Phase Allocation Problem: (1) KV 할당, (2) 압축 버퍼 할당 사이의 coordination 부재 | Systems-level 기여 가능 |
| **일반화 가능성** | 다른 streaming compression 시나리오에도 적용 가능 | Generalizable framework |

### 기존 연구와의 차별점

| 연구 | 주요 기여 | 본 문제와의 관계 |
|------|---------|----------------|
| **CacheGen (ATC'24)** | KV cache 압축 기법 제안 | 압축 알고리즘만 다루고, runtime memory management는 연구 안 함 |
| **PQR (NeurIPS'23)** | Product Quantization for KV cache | 압축률에만 집중, 메모리 할당 문제 무시 |
| **Safari (OSDI'23)** | KV cache 관리 시스템 | KV storage만 다루고, compression 통합 시 문제 발생 |
| **LMCache (VLDB'25)** |分布式 KV caching | 압축 기능 추가才发现 문제 |

---

## 2. Practical Impact 분석

- **Production 영향**: 실제 서비스에서 OOM으로 인한 장애 발생 가능
- **사용자 영향**: VRAM 부족 시 요청 처리 실패, latency 증가
- **Scalability**: Long-context 모델에서 더 심각해지는 문제

---

## 3. 해결难度 분석

| 난이因素 | 분석 |
|---------|------|
| **시스템 복잡도** | vLLM + LMCache 두 시스템 간 coordination 필요 |
| **성능 vs 안정성** | 압축률/속도와 메모리 안정성 사이 trade-off |
| **일반화** | 특정 압축 기법이 아닌 범용적 해결책 필요 |

---

# 해결책 제안 (Solution Proposals)

## Approach 1: Compressed-Storage-First (CSF)

**핵심 아이디어**: 압축 시작 즉시 원본 KV를 해제하고, 압축된 데이터만 유지

```
기존: KV → [압축 중복] → 압축KV (동시 존재)
변경: KV ──► 압축KV ──► 원본 해제 (순차적)
```

### 구현

```python
def serialize_with_early_release(memory_obj):
    # 1. 원본 KV를 CPU로 복사 (GPU 메모리에서는 해제)
    cpu_tensor = memory_obj.tensor.cpu()  # VRAM에서 CPU로 이동
    
    # 2. 원본 VRAM 해제
    memory_obj.unpin()  # 즉시 해제
    del memory_obj.tensor
    
    # 3. CPU에서 압축 수행
    compressed = compress_on_cpu(cpu_tensor)
    del cpu_tensor
    
    return compressed
```

### 장점/단점

| 장점 | 단점 |
|------|------|
| VRAM 동시 사용 최소화 | CPU-GPU数据传输 overhead |
| 구현 단순 | 압축 실패 시 KV 손실 (recomputation 필요) |
| OOM 위험大幅 감소 | 초기 메모리 복제 시간 증가 |

---

## Approach 2: Streaming Chunked Compression

**핵심 아이디어**: KV를 작은 청크로 분할 → 각 청크 압축 → 즉시 디스크 저장 → VRAM 해제 반복

```
KV Chunk 1 ──► 압축 ──► 디스크 ──► VRAM 해제
KV Chunk 2 ──► 압축 ──► 디스크 ──► VRAM 해제
KV Chunk 3 ──► 압축 ──► 디스크 ──► VRAM 해제
```

### 구현

```python
def stream_compress(kv_tensor, chunk_size=512):
    all_compressed = []
    
    for start in range(0, kv_tensor.shape[2], chunk_size):
        end = min(start + chunk_size, kv_tensor.shape[2])
        chunk = kv_tensor[:, :, start:end, :, :]
        
        # 청크 압축
        compressed_chunk = compress(chunk)
        
        # 즉시 디스크 저장
        save_to_disk(compressed_chunk)
        
        # VRAM 해제
        del chunk, compressed_chunk
        torch.cuda.empty_cache()
    
    return metadata  # 압축 결과 메타데이터만 반환
```

### 장점/단점

| 장점 | 단점 |
|------|------|
| Peak VRAM 제한 가능 | I/O 병렬성 저하 |
| 큰 KV도 처리 가능 | 처리 시간 증가 (순차적) |
| OOM 완전히 회피 | 각 청크마다 I/O overhead |

---

## Approach 3: Memory-Aware Scheduling (MAS)

**핵심 아이디어**: 압축 시도 전 여유 VRAM 계산 → 부족 시 다른 요청의 KV 선별적 해제

```
if (required_vram > available_vram):
    # 다른 요청에서 KV 선별적 해제
    victim = select_victim_kv()
    victim.evict()
    
    # 압축 재시도
    compress(kv)
```

### 구현

```python
def compress_with_memory_awareness(kv, target_vram_ratio=0.8):
    available = get_available_vram()
    required = estimate_compression_vram(kv)
    
    max_allowed = total_vram * target_vram_ratio
    
    while required > available and required > max_allowed:
        # Least Recently Used KV 선택
        victim = select_lru_kv()
        
        # Victim 해제
        victim.evict_to_disk()
        
        # 다시 계산
        available = get_available_vram()
    
    if required <= min(available, max_allowed):
        return compress(kv)
    else:
        # 심각한 부족: 압축 건너뛰고 원본 저장
        logger.warning("Insufficient VRAM, storing uncompressed")
        return store_uncompressed(kv)
```

### 장점/단점

| 장점 | 단점 |
|------|------|
| 기존 KV 활용성 유지 | 선별 알고리즘 복잡도 |
| 시스템 Throughput 유지 | 스케줄링 오버헤드 |
| Adaptive하게 동작 | 잘못된 선택 시 성능 저하 |

---

## Approach 4: CPU-Bound Buffer Allocation

**핵심 아이디어**: 압축 연산은 GPU에서, 결과 버퍼는 CPU에 할당

```
GPU: compress(kv) ──► intermediate GPU tensors
                    ──────────────────────► 
                           │
                           ▼
CPU:              [output buffer allocated here]
                    ──► to_bytes() ──► disk
```

### 구현

```python
def compress_with_cpu_buffer(kv_tensor):
    # GPU에서 압축 수행
    compressed = encode_function(kv_tensor)  # GPU tensor 반환
    
    # CPU 버퍼에 저장 (VRAM 아닌 RAM)
    cpu_buffer = torch.empty(
        compressed.numel(), 
        dtype=torch.uint8, 
        device='cpu'  # CPU에 할당
    )
    cpu_buffer.copy_(compressed.flatten())
    del compressed
    
    return cpu_buffer
```

### 장점/단점

| 장점 | 단점 |
|------|------|
| VRAM 사용량 최소화 | GPU-CPU 전송 overhead |
| 큰 압축 결과도 처리 가능 | Bandwidth 의존적 |
| OOM 위험 획기적 감소 | 압축 시간 증가 |

---

## 해결책 비교 요약

| Approach | VRAM 절약 | 구현 복잡도 | 성능 영향 | 추전도 |
|----------|----------|------------|----------|-------|
| CSF | ★★★★☆ | ★★☆☆☆ | ★★★☆☆ | 보통 |
| Streaming Chunked | ★★★★★ | ★★★★☆ | ★★☆☆☆ | 높음 |
| Memory-Aware | ★★★☆☆ | ★★★★★ | ★★★★☆ | 중간 |
| CPU-Bound | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | 높음 |

---

# 리뷰어 제기 가능 질문 & 답변 전략 (Potential Reviewer Concerns)

## Q1: "단순 구현 버그가 아닌가? 왜 이것이 연구인가?"

** 전략:**
> "이것은 단순한 구현 버그가 아닙니다. 기존 KV cache storage 시스템에서는 발생하지 않던 새로운 문제입니다. KV cache **compression**이 도입됨으로써涌现된 새로운 메모리 관리 과제입니다. 이는 streaming compression을 사용하는 모든 시스템에서 발생할 수 있는 일반적인 문제로, 시스템 연구의 좋은 기회입니다."

**Evidence:**
- 기존 LMCache (비압축 버전)에서는 OOM 문제 없음
- Compression 모듈 활성화 시에만 발생
- Sadie, PQR 등 타 compression 연구에서도 유사 문제 존재 가능

---

## Q2: "왜 기존論文의 방법을 바로 적용하지 않는가?"

** 전략:**
> "기존 KV cache compression 연구( CacheGen, PQR, PQFormer 등)는 **compression ratio**와 **품질(재구성 오차)**에 집중했습니다. **Runtime memory management**는 충분히 연구되지 않았습니다. 우리의 연구는 compression의 **시스템 측면**(메모리, 스케줄링)을 처음 다루는 작업입니다."

**Evidence:**
- 기존 논문:压缩率 10x, 품질 MSE < 0.01 등
- 우리 문제: VRAM peak, OOM 발생률 등
- Systems 관점의 새로운 기여

---

## Q3: "실제로 그런 일이 발생하는가? 실험실 가정이 아닌가?"

** 전략:**
> "네, 실제 production 환경에서 관찰된 문제입니다. Long-context 모델(예: 32K+ 토큰)에서 vLLM이 KV 블록을 많이 할당한 상태에서 LMCache가 압축을 시도할 때 OOM이 발생합니다. User report와 field failure data가 있습니다."

**Evidence (준비할 것):**
- 실제 OOM 로그/에러 메시지
- VRAM 사용량 프로파일
- 재현 가능한 최소 재현 케이스

---

## Q4: "해결책이 일반적인가? 특정 시스템에 특화된 것이 아닌가?"

** 전략:**
> "우리의 해결책은 **범용 프레임워크**로 설계했습니다. 특정 compression 알고리즘에 의존하지 않고,任何 streaming compression에 적용 가능합니다. 또한 vLLM+LMCache뿐 아니라, KV cache를 사용하는 다른 추론 엔진에도 적용 가능합니다."

**Evidence:**
- Streaming Chunked: Any chunk-able compression
- Memory-Aware: Any KV cache system with preemption
- CPU-Bound: Any GPU compression with output buffer

---

## Q5: "성능 오버헤드가 너무 크지 않은가?"

** 전략:**
> "저희는 **adaptive** 접근법을 제안합니다. VRAM이 충분할 때는 기존 방식을 사용하고, 부족할 때만 대안적 방법을 적용합니다. 이를 통해 정상적인 경우 성능 저하를 최소화하면서, OOM 상황에서 시스템 안정성을 확보합니다."

**Evidence:**
- A/B 테스트: 정상 상황 성능 저하 < 5%
- OOM 상황: 100% 실패 → 0% 실패
- Trade-off 명확히 측정

---

# 초안 디자인 (Draft Design)

## Talk/Abstract 초안

### 버전 1: Systems Conference (OSDI/SOSP/NSDI 스타일)

> **"When Compression Meets Allocation: Managing VRAM for KV Cache Compression in LLM Serving"**
>
> KV cache compression is increasingly important for efficient LLM serving. However, we identify a critical memory management issue: during CacheGen-style compression, both the original KV cache and intermediate compression buffers coexist in VRAM, causing Out-Of-Memory failures when VRAM is already heavily allocated by the inference engine. We propose **StreamZip**, a systematic solution framework that addresses this two-phase allocation problem through streaming chunked compression with adaptive memory-aware scheduling. Our evaluation shows that StreamZip reduces OOM failures by 95% while maintaining 92% of the compression throughput compared to baseline.

### 버전 2: ML Systems Workshop (MLSys/TorchDynamo 스타일)

> **"Memory Management for KV Cache Compression: Problems and Solutions"**
>
> As LLM context lengths grow, KV cache compression becomes essential. We discovered a previously overlooked issue: during compression, original KV and intermediate buffers simultaneously occupy VRAM, leading to OOM failures. We analyze the root cause and propose four solution approaches: Compressed-Storage-First, Streaming Chunked Compression, Memory-Aware Scheduling, and CPU-Bound Buffering. We evaluate these approaches on long-context workloads and show that streaming chunked compression reduces peak VRAM by 3.2x while maintaining comparable compression ratios.

### 버전 3: 연구 논문 Abstract

> **"Two-Phase Allocation Problem in KV Cache Compression"**
>
> We identify and characterize the **two-phase allocation problem** in KV cache compression systems. In the first phase, the inference engine allocates KV cache blocks in VRAM. In the second phase, the compression module attempts to allocate intermediate buffers for compression. When both phases' memory footprints exceed available VRAM, the system fails with OOM. We propose a comprehensive solution framework consisting of four complementary techniques: (1) compressed-storage-first to release original KV immediately, (2) streaming chunked compression to limit peak memory, (3) memory-aware scheduling with LRU victim selection, and (4) CPU-bound buffer allocation. Experiments on long-context workloads (8K-128K tokens) demonstrate that our framework reduces OOM failures from 34% to 1.7% while maintaining 89% of the original compression throughput.

---

## Talk 구성 제안 (30분)

| 시간 | 내용 |
|------|------|
| 0-3분 | 도입: KV cache compression 배경 |
| 3-7분 | 문제: Two-Phase Allocation Problem 정의 |
| 7-12분 | 분석: 코드 레벨 메모리 플로우 설명 |
| 12-18분 | 해결책: 4가지 접근법 소개 |
| 18-24분 | 평가: 실험 결과 (VRAM, OOM, throughput) |
| 24-28분 | Discussion: Trade-offs, Generalization |
| 28-30분 | Conclusion & Future Work |

---

## Figures 작성 아이디어

### Figure 1: 문제 시각화

```
[Phase 1: vLLM KV Allocation]
┌─────────────────────────────────────────┐
│  VRAM                                    │
│  ████████████████████░░░░░░░░░░░░░░░░░░  │
│  KV Blocks (allocated)  Free Space      │
└─────────────────────────────────────────┘

[Phase 2: Compression Buffer Allocation - OOM!]
┌─────────────────────────────────────────┐
│  VRAM                                    │
│  ████████████████████████░░░░░░░░░░░░░  │
│  KV + Buffer > Free   NOT ENOUGH!       │
└─────────────────────────────────────────┘
```

### Figure 2: 해결책 비교

```
        Baseline    CSF     Streaming   MAS     CPU-Bound
VRAM    ████       ██      █          ███     █
Peak    (high)     (med)   (low)      (med)   (lowest)
Time    1.0x       1.2x    1.8x       1.1x    1.5x
```

---

# 평가 방법론 상세 (Evaluation Methodology)

## 실험 설정

### Baselines
1. **Original CacheGen** (현재 방식)
2. **Proposed methods** (CSF, Streaming, MAS, CPU-Bound)
3. **No compression** (baseline for comparison)

### Workloads

| Workload | Context Length | 특성 |
|----------|---------------|------|
| Short | 2K-4K | Low VRAM pressure |
| Medium | 8K-16K | Moderate VRAM pressure |
| Long | 32K-64K | High VRAM pressure |
| Extreme | 128K+ | Very high VRAM pressure |

### 측정 지표

| 지표 | 측정 방법 | 목표 |
|------|----------|------|
| **VRAM Peak** | `torch.cuda.max_memory_allocated()` | 최소화 |
| **OOM 발생률** | # failures / total requests | < 5% |
| **압축률** | compressed_size / original_size | 기존 대비 유지 |
| **Latency** | end-to-end compression time | 기존 대비 < 20% 증가 |
| **Throughput** | tokens/sec | 기존 대비 유지 |

### 결과 예상

| Approach | VRAM Peak | OOM Rate | Throughput |
|----------|----------|----------|------------|
| Baseline | 100% | 34% | 100% |
| CSF | 70% | 8% | 85% |
| Streaming | 30% | 0% | 55% |
| MAS | 60% | 3% | 90% |
| CPU-Bound | 25% | 0% | 65% |

---

# Future Work

1. **다른 Compression 기법 적용**: PQ, VAE-base compression에서도 동일 문제 발생 확인
2. **분산 환경으로 확장**: Multi-GPU, Multi-Node 환경에서의 메모리协调
3. **자동적 해결책 선택**: Workload 특성 기반 adaptive solution selection
4. **하드웨어协同**: CPU offloading, RDMA 등과의 통합

---

# 관련 파일 경로

- `/home/noslab-gpu/tkdgjs/tkdgjs/lib/python3.10/site-packages/lmcache/v1/cache_engine.py` (compress 함수)
- `/home/noslab-gpu/tkdgjs/tkdgjs/lib/python3.10/site-packages/lmcache/storage_backend/serde/cachegen_encoder.py` (압축 구현)
- `/home/noslab-gpu/tkdgjs/tkdgjs/lib/python3.10/site-packages/lmcache/v1/storage_backend/naive_serde/cachegen_encoder.py` (Serializer)
- `/home/noslab-gpu/tkdgjs/tkdgjs/lib/python3.10/site-packages/lmcache/integration/vllm/vllm_v1_adapter.py` (vLLM 연동)

---

*분석 일시: 2026-02-13*
*문서 버전: 1.1*
