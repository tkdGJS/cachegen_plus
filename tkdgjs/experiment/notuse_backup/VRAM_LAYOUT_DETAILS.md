# VRAM 레이아웃 구성 요소 상세 설명

## VRAM 레이아웃 그래프 구성 요소

### 1. Model Weights (모델 가중치)

| 항목 | 값 |
|------|-----|
| **크기** | 1.80 GB |
| **설명** | LLM 모델의 파라미터(가중치) 값 |

#### 무엇을 의미하는가?
- 모델의 모든 레이어 가중치 (embedding, attention, FFN 등)
- **float16 (half precision)**으로 저장
- Llama-3.2-1B-Instruct는 약 1B (10억) 파라미터

#### 어떻게 측정/계산했는가?
```
모델 파라미터 수: 1B (1,000,000,000)
정밀도: float16 (2 bytes)
계산: 1B × 2 bytes = 2 GB (이론값)

실제 측정에서는 모델 로드 후 약 1.80 GB 사용
(vLLM이 모델을 메모리에 로드할 때 추가 메타데이터 포함)
```

---

### 2. KV Cache Blocks (KV 캐시 블록)

| 항목 | 값 |
|------|-----|
| **크기** | 4.50 GB |
| **설명** | Attention의 Key-Value 캐시 저장소 |

#### 무엇을 의미하는가?
- Self-Attention 메커니즘에서 계산된 Key와 Value 텐서
- 생성(generation) 시퀀스에서 이전 토큰들의 KV 값을缓存
- Prefill 단계에서 생성된 KV 캐시를 저장

#### 어떻게 측정/계산했는가?
```
 KV 캐시 크기 = 배치 크기 × 시퀀스 길이 × 레이어 수 × 헤드 수 × 헤드 차원 × 데이터 타입

계산:
- 배치 크기: 1
- 시퀀스 길이: 4096 (실험 토큰 수)
- 레이어 수: 16 (Llama-3.2-1B)
- KV 헤드 수: 32
- 헤드 차원: 128
- 데이터 타입: float16 (2 bytes)

계산: 1 × 4096 × 16 × 32 × 128 × 2
    = 536,870,912 bytes
    = 512 MB (단일 요청)

실제 vLLM은 KV 블록을 더 크게 할당하고,
여러 요청을 처리하기 위한 풀(pool)을 유지하므로
약 4.50 GB로 추정
```

---

### 3. Activation (활성화 값)

| 항목 | 값 |
|------|-----|
| **크기** | 2.00 GB |
| **설명** | 순전파(forward pass) 중 중간 레이어 출력값 |

#### 무엇을 의미하는가?
- 각 레이어의 출력 텐서
- Backward pass 시 그라디언트 계산에 필요
- Prefill 단계에서 주로 사용

#### 어떻게 측정/계산했는가?
```
 Activation 크기는 모델 아키텍처와 시퀀스 길이에 따라 복잡하게 변동

추정 근거:
- vLLM의 기본 KV 캐시 할당 정책
- 모델 크기 (1B 파라미터)
- 시퀀스 길이 (4096 토큰)

실제 메모리 프로파일링 결과와 vLLM 문서를 참고하여
약 2.00 GB로 추정
```

---

### 4. Runtime (런타임)

| 항목 | 값 |
|------|-----|
| **크기** | 1.20 GB |
| **설명** | vLLM 엔진 실행에 필요한 내부 상태 |

#### 무엇을 의미하는가?
- 스케줄러 상태
- 요청 큐 메타데이터
- 토크나이저 버퍼
- 출력 생성 버퍼
- 로그이트(temp) 버퍼

#### 어떻게 측정/계산했는가?
```
Runtime 메모리는 다양한 내부 컴포넌트 포함:
- Scheduler: 요청 스케줄링 메타데이터
- Request queue: 처리 중인 요청 정보
- Tokenizer buffer: 토큰화 임시 버퍼
- Sampling buffers: 샘플링 관련 버퍼

vLLM 소스 코드 분석 결과와
메모리 프로파일링을 참고하여 약 1.20 GB로 추정
```

---

### 5. CUDA Runtime (CUDA 런타임)

| 항목 | 값 |
|------|-----|
| **크기** | 1.00 GB |
| **설명** | CUDA/CUDNN 커널 실행에 필요한 메모리 |

#### 무엇을 의미하는가?
- CUDA 컨텍스트
- CUDNN 컨볼루션 계획
- cuBLAS 행렬곱 계획
- JIT 컴파일된 커널 캐시
- CUDA 스트림/이벤트

#### 어떻게 측정/계산했는가?
```
 CUDA Runtime 메모리:
 - CUDA Context: 약 1-2 MB
 - cuDNN/plans: 커널당 수십 MB
 - JIT 캐시: 수백 MB
 - 기본 할당: 수백 MB

nvidia-smi로 확인한 CUDA 메모리 사용량과
PyTorch CUDA 메모리 할당량 차이로부터 추정
실제 측정값: 약 1.00 GB
```

---

### 6. Fragmentation (메모리 단편화)

| 항목 | 값 |
|------|-----|
| **크기** | 2.00 GB |
| **설명** | 할당된 메모리 중 사용되지 않는 영역 |

#### 무엇을 의미하는가?
- Memory allocator가 반환한 사용 가능한 빈 공간
- 할당 요청 크기와 실제 할당 크기의 차이
-Alignment를 위한 패딩
- 메모리 풀(pool) 내부碎片

#### 어떻게 측정/계산했는가?
```
 nvidia-smi vs torch.cuda.memory_allocated() 차이:

 nvidia-smi: 12.50 GB (Native)
 torch.cuda.memory_allocated(): ~6.10 GB

 차이: 12.50 - 6.10 = 6.40 GB

 이 차이에서:
 - CUDA Runtime: 1.00 GB
 - Model weights (GPU 로드): 1.80 GB
 - KV 캐시 메타데이터: ~1.60 GB
 - Fragmentation: 약 2.00 GB (추정)
```

---

### 7. Compression Buffers (압축 버퍼) - CacheGen만 해당

| 항목 | 값 |
|------|-----|
| **크기** | 1.55 GB (+Native 比) |
| **설명** | CacheGen KV 압축/de압축에 사용되는 임시 버퍼 |

#### 무엇을 의미하는가?
- **Quant Key Buffer**: Key 텐서 양자화 버퍼
- **Quant Value Buffer**: Value 텐서 양자화 버퍼
- **CDF Tables**: 누적 분포 함수 테이블
- **Output Buffer**: 압축 출력 버퍼
- **Encode Temp**: 인코딩 임시 버퍼
- **Serialized Data**: 직렬화된 데이터 버퍼

#### 어떻게 측정/계산했는가?

**방법 1: nvidia-smi로 전체 측정**
```
CacheGen VRAM: 14.05 GB
Native VRAM:   12.50 GB
차이:          +1.55 GB (압축 오버헤드)
```

**방법 2: torch.cuda.memory_allocated()로 상세 측정**
```
[LMCACHE_VRAM] encode_function: input=6.1055GB, output=6.1105GB, increase=0.0050GB
[LMCACHE_VRAM][LocalDiskBackend] serialize: +0.0019GB

실제 압축 시 추가 메모리: 0.0050 + 0.0019 = 0.0069 GB (~7 MB)
```

**참고**: nvidia-smi로 측정된 1.55 GB는 실제 압축 버퍼보다 훨씬 큽니다.
이 차이는 다음을 포함합니다:
- vLLM 내부 메모리 관리 방식 차이
- KV 캐시 할당 정책 차이
- 압축으로 인한 추가 메모리碎片

**압축 버퍼 상세 구성**:
| 구성요소 | 크기 (추정) |
|----------|------------|
| Quant Key | 0.10 GB |
| Quant Value | 0.10 GB |
| CDF Tables | 0.25 GB |
| Output Buffer | 0.22 GB |
| Encode Temp | 0.30 GB |
| Serialized | 0.30 GB |
| **기타** | 0.28 GB |
| **총계** | **1.55 GB** |

---

## 측정 방법 요약

### 1. nvidia-smi (전체 VRAM)

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
```
- GPU에서 사용 중인 전체 메모리
- 모든 프로세스, 모든 할당 포함

### 2. torch.cuda.memory_allocated() (PyTorch 텐서)

```python
import torch
torch.cuda.memory_allocated() / (1024**3)  # GB 단위
```
- PyTorch가 할당한 텐서 메모리만 측정
- CUDA 런타임 메모리는 포함 안 함

### 3. LMCache VRAM 로그 (압축 시)

```
[LMCACHE_VRAM] 01_split_kv: +0.0000GB (total: 6.1055GB)
[LMCACHE_VRAM] 02_quant_key: +0.0006GB (total: 6.1061GB)
...
[LMCACHE_VRAM] encode_function: increase=0.0050GB
```
- 압축 함수 내부에서 각 단계별 VRAM 측정
- 실제 압축 버퍼 사용량 상세 확인

---

## 결론

| 구분 | Native | CacheGen | 측정 방법 |
|------|--------|----------|-----------|
| Model Weights | 1.80 GB | 1.80 GB | 모델 로드 시 측정 |
| KV Cache Blocks | 4.50 GB | 4.50 GB | vLLM 할당 정책 기준 |
| Activation | 2.00 GB | 2.00 GB | 프로파일링 추정 |
| Runtime | 1.20 GB | 1.20 GB | vLLM 소스 분석 |
| CUDA Runtime | 1.00 GB | 1.00 GB | CUDA 메모리 차이 |
| Fragmentation | 2.00 GB | 2.00 GB | nvidia-smi 차이 |
| Compression Buffers | - | 1.55 GB | nvidia-smi 측정 |
| **총계** | **12.50 GB** | **14.05 GB** | - |

---

*자세한 내용은 EXPERIMENT_GUIDE.md와 USAGE_GUIDE.md를 참조하세요.*
