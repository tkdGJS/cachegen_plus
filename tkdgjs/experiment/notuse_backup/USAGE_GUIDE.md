# LMCache VRAM 실험 - 사용 가이드

## 실험 개요

이 실험은 LMCache의 **CacheGen 압축 모드**와 **Native (Torch) 비압축 모드**의 VRAM 사용량을 비교 분석합니다.

### 핵심 결과

| 구분 | Native (Torch) | CacheGen | 차이 |
|------|---------------|----------|------|
| **VRAM (nvidia-smi)** | 12.50 GB | 14.05 GB | +1.55 GB |
| **압축 버퍼 (torch.cuda.memory)** | 0 GB | ~7 MB | +7 MB |
| **디스크 저장** | ~7.9 MB | 1.1 MB | 7.2x 압축 |

---

## 1. 필요한 파일 목록

### 1.1 설정 파일

| 파일 | 설명 |
|------|------|
| `lmcache_cachegen.yaml` | CacheGen (압축) 모드 설정 |
| `lmcache_native.yaml` | Native (비압축) 모드 설정 |

### 1.2 실행 스크립트

| 파일 | 설명 |
|------|------|
| `generate_vram_verified.py` | VRAM 분석 그래프 생성 |

### 1.3 수정된 LMCache 코드

| 파일 | 설명 |
|------|------|
| `lmcache/v1/storage_backend/local_disk_backend.py` | VRAM 모니터링 및 압축 적용 코드 |

---

## 2. 실험 실행 순서

### Step 1: 환경 정리

```bash
# 기존 vLLM 프로세스 종료
pkill -9 -f vllm 2>/dev/null || true
sleep 3

# GPU 메모리 확인
nvidia-smi
```

**동작**: 기존에 실행 중인 vLLM 프로세스를 모두 종료하고 GPU 메모리 상태를 확인합니다.

---

### Step 2: CacheGen 모드 VRAM 측정

```bash
# 1) 환경 변수 설정
export LMCACHE_CONFIG_FILE=/home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen.yaml
export LMCACHE_VRAM_LOG=1
export LMCACHE_VRAM_LOG_FILE=/tmp/lmcache_vram.log
export VLLM_ATTENTION_BACKEND=TRITON_ATTN

# 2) 캐시 디렉토리 초기화
rm -rf /home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen_disk/*
rm -f /tmp/lmcache_vram.log

# 3) vLLM 시작
/home/noslab-gpu/tkdgjs/tkdgjs/bin/vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --port 8005 \
  --dtype half \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.5 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"discard_partial_chunks":false}}'
```

**동작**:
- `LMCACHE_CONFIG_FILE`: 사용할 LMCache 설정 파일 지정 (CacheGen)
- `LMCACHE_VRAM_LOG=1`: VRAM 모니터링 활성화
- `VLLM_ATTENTION_BACKEND=TRITON_ATTN`: NVCC 오류 회피

---

### Step 3: CacheGen 요청 전송 및 측정

```bash
# 1) vLLM 준비 확인 (새 터미널)
curl -s http://localhost:8005/v1/models

# 2) Inference 요청 전송
curl -s http://localhost:8005/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "prompt": "Write a detailed story about a dragon.",
    "max_tokens": 200,
    "temperature": 0.7
  }'

# 3) VRAM 측정
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

# 4) LMCache VRAM 로그 확인
cat /tmp/lmcache_vram.log

# 5) 디스크 캐시 크기 확인
ls -la /home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen_disk/*.pt
```

**동작**:
- Inference 요청을 보내 KV 캐시 생성 및 압축 저장
- `nvidia-smi`: 전체 GPU 메모리 사용량 측정
- `cat /tmp/lmcache_vram.log`: 압축 시 VRAM 변화 상세 로그
- `ls -la *.pt`: 압축된 캐시 파일 크기 확인

---

### Step 4: Native 모드 VRAM 측정

```bash
# 1) vLLM 종료
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 3

# 2) 환경 변수 변경 (Native 설정)
export LMCACHE_CONFIG_FILE=/home/noslab-gpu/tkdgjs/experiment/lmcache_native.yaml
export LMCACHE_VRAM_LOG=1
export LMCACHE_VRAM_LOG_FILE=/tmp/lmcache_vram.log

# 3) 캐시 디렉토리 초기화
rm -rf /home/noslab-gpu/tkdgjs/experiment/lmcache_torch_disk/*
rm -f /tmp/lmcache_vram.log

# 4) vLLM 시작 (Native 모드)
/home/noslab-gpu/tkdgjs/tkdgjs/tkdgjs/bin/vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --port 8005 \
  --dtype half \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.5 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"discard_partial_chunks":false}}'
```

**동작**: 설정 파일만 `lmcache_native.yaml`로 변경하여 Native 모드로 재실행

---

### Step 5: Native 요청 전송 및 측정

```bash
# 1) Inference 요청
curl -s http://localhost:8005/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "prompt": "Write a detailed story about a dragon.",
    "max_tokens": 200,
    "temperature": 0.7
  }'

# 2) VRAM 측정
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

# 3) 디스크 캐시 확인
ls -la /home/noslab-gpu/tkdgjs/experiment/lmcache_torch_disk/*.pt
```

---

### Step 6: 그래프 생성

```bash
# VRAM 분석 그래프 생성
/home/noslab-gpu/tkdgjs/tkdgjs/bin/python /home/noslab-gpu/tkdgjs/experiment/generate_vram_verified.py
```

**동작**: 측정 데이터를 기반으로 VRAM Breakdown 그래프 생성
- 생성된 파일: `/tmp/vram_verified_breakdown.png`

---

### Step 7: 이메일 전송 (선택)

```bash
# 이메일 전송 스크립트 사용
python3 /home/noslab-gpu/tkdgjs/experiment/send_email.py
```

또는 수동:

```python
import email.mime.multipart
import email.mime.image
import smtplib

msg = email.mime.multipart.MIMEMultipart()
msg['Subject'] = '[VRAM Analysis] Results'
msg['From'] = 'tkdgjs0213@gmail.com'
msg['To'] = 'tkdgjs0213@gmail.com'

with open('/tmp/vram_verified_breakdown.png', 'rb') as f:
    img = email.mime.image.MIMEImage(f.read(), 'png')
    img.add_header('Content-Disposition', 'attachment', filename='vram_breakdown.png')
    msg.attach(img)

server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('tkdgjs0213@gmail.com', 'app_password')
server.send_message(msg)
server.quit()
```

---

## 3. 명령어 상세 설명

### 3.1 환경 변수

| 변수 | 값 | 설명 |
|------|-----|------|
| `LMCACHE_CONFIG_FILE` | 설정 파일 경로 | 사용할 LMCache 설정 |
| `LMCACHE_VRAM_LOG` | 1 | VRAM 모니터링 활성화 |
| `LMCACHE_VRAM_LOG_FILE` | 파일 경로 | VRAM 로그 저장 위치 |
| `VLLM_ATTENTION_BACKEND` | TRITON_ATTN | Attention 백엔드 지정 (NVCC 오류 회피) |

### 3.2 vLLM 실행 옵션

| 옵션 | 값 | 설명 |
|------|-----|------|
| `--port` | 8005 | vLLM API 서버 포트 |
| `--dtype` | half | 데이터 타입 (float16) |
| `--max-model-len` | 8192 | 최대 모델 시퀀스 길이 |
| `--gpu-memory-utilization` | 0.5 | GPU 메모리 사용률 (50%) |
| `--kv-transfer-config` | JSON | KV 전송 설정 |

### 3.3 KV 전송 설정 (kv-transfer-config)

```json
{
  "kv_connector": "LMCacheConnectorV1",  # LMCache 커넥터 사용
  "kv_role": "kv_both",                   # KV 저장 및 로드 모두
  "kv_connector_extra_config": {
    "discard_partial_chunks": false       # 부분 청크 버리지 않음
  }
}
```

### 3.4 측정 명령어

| 명령어 | 설명 |
|--------|------|
| `nvidia-smi` | GPU 상태 전체 확인 |
| `nvidia-smi --query-gpu=memory.used` | GPU 메모리 사용량 (MB) |
| `cat /tmp/lmcache_vram.log` | LMCache VRAM 상세 로그 |
| `ls -la *.pt` | 캐시 파일 크기 확인 |

---

## 4. LMCache 설정 파일 비교

### 4.1 CacheGen (lmcache_cachegen.yaml)

```yaml
chunk_size: 256
save_unfull_chunk: true
local_cpu: true
max_local_cpu_size: 4.0
local_disk: "file:///home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen_disk/"
max_local_disk_size: 10.0
remote_serde: cachegen  # ← 압축 사용
```

### 4.2 Native (lmcache_native.yaml)

```yaml
chunk_size: 256
save_unfull_chunk: true
local_cpu: true
max_local_cpu_size: 4.0
local_disk: "file:///home/noslab-gpu/tkdgjs/experiment/lmcache_torch_disk/"
max_local_disk_size: 10.0
remote_serde: torch  # ← 압축 없음
```

---

## 5. 예상 결과

### CacheGen 모드

```
VRAM (nvidia-smi): ~14.05 GB
VRAM 로그:
  [LMCACHE_VRAM] encode_function: increase=0.0050GB
  [LMCACHE_VRAM][LocalDiskBackend] serialize: +0.0019GB
디스크 파일 크기: ~1.1 MB (압축됨)
```

### Native 모드

```
VRAM (nvidia-smi): ~12.50 GB
VRAM 로그: (압축 없음)
디스크 파일 크기: ~7.9 MB (원본)
```

---

## 6. 문제 해결

### NVCC 오류 발생 시

```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

**해결**: `VLLM_ATTENTION_BACKEND=TRITON_ATTN` 환경 변수 추가

### GPU 메모리 부족 오류 발생 시

```
ValueError: Free memory on device cuda:0 (X.XX/14.56 GiB) is less than desired GPU memory utilization
```

**해결**: `--gpu-memory-utilization` 값을 낮춤 (예: 0.3)

### 포트 사용 중 오류 발생 시

```
OSError: [Errno 98] Address already in use
```

**해결**: 다른 포트 사용 (예: `--port 8006`)

---

## 7. 파일 위치 요약

| 구분 | 경로 |
|------|------|
| 설정 파일 | `/home/noslab-gpu/tkdgjs/experiment/lmcache_*.yaml` |
| VRAM 로그 | `/tmp/lmcache_vram.log` |
| 디스크 캐시 (CacheGen) | `/home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen_disk/` |
| 디스크 캐시 (Native) | `/home/noslab-gpu/tkdgjs/experiment/lmcache_torch_disk/` |
| 결과 그래프 | `/tmp/vram_verified_breakdown.png` |
| 가이드 문서 | `/home/noslab-gpu/tkdgjs/experiment/EXPERIMENT_GUIDE.md` |

---

*생성일: 2026-02-16*
