# LMCache VRAM 분석 실험 - 종합 가이드

## 개요

이 프로젝트는 LMCache의 CacheGen 압축 모드와 Native (Torch) 모드의 VRAM 사용량을 비교 분석합니다.

### 주요 발견

| 구분 | Native (Torch) | CacheGen |
|------|---------------|----------|
| **VRAM (nvidia-smi)** | 12.50 GB | 14.05 GB |
| **추가 VRAM** | - | +1.55 GB |
| **디스크 저장** | ~7.9 MB | 1.1 MB |
| **압축률** | - | 7.2x |

---

## 1. 실험 환경

### 하드웨어
- **GPU**: Tesla T4 (14.56 GB VRAM)
- **CPU**: AMD EPYC or similar
- **RAM**: 64 GB+

### 소프트웨어
- **Python**: 3.10.19
- **vLLM**: 0.14.0
- **LMCache**: Custom modified version
- **CUDA**: 12.8
- **Attention Backend**: TRITON_ATTN

---

## 2. 프로젝트 구조

```
/home/noslab-gpu/tkdgjs/experiment/
├── lmcache_cachegen.yaml       # CacheGen 설정 (압축)
├── lmcache_native.yaml         # Native 설정 (비압축)
├── lmcache_cachegen_disk/      # CacheGen KV 캐시 저장소
├── lmcache_torch_disk/        # Native KV 캐시 저장소
├── monitor_vram.py            # VRAM 모니터링 스크립트
├── generate_vram_verified.py  # 검증된 VRAM 그래프 생성
├── send_email.py               # 이메일 전송
└── vram_verified_breakdown.png # 결과 그래프
```

---

## 3. LMCache 설정 파일

### lmcache_cachegen.yaml (압축 모드)
```yaml
chunk_size: 256
save_unfull_chunk: true
local_cpu: true
max_local_cpu_size: 4.0
local_disk: "file:///home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen_disk/"
max_local_disk_size: 10.0
remote_serde: cachegen  # ← 압축 사용
enable_async_loading: true
enable_kv_events: true
internal_api_server_enabled: true
internal_api_server_port_start: 6999
```

### lmcache_native.yaml (비압축 모드)
```yaml
chunk_size: 256
save_unfull_chunk: true
local_cpu: true
max_local_cpu_size: 4.0
local_disk: "file:///home/noslab-gpu/tkdgjs/experiment/lmcache_torch_disk/"
max_local_disk_size: 10.0
remote_serde: torch  # ← 압축 없음
enable_async_loading: true
enable_kv_events: true
internal_api_server_enabled: true
internal_api_server_port_start: 6999
```

---

## 4. 코드 수정 (VRAM 모니터링)

### 수정된 파일

**파일 경로**: `/home/noslab-gpu/tkdgjs/tkdgjs/lib/python3.10/site-packages/lmcache/v1/storage_backend/local_disk_backend.py`

### 수정 내용 1: __init__에 serializer 초기화 추가 (line ~154)
```python
# Initialize serializer for compression (CacheGen or Torch/Naive)
self.serializer = None
self.deserializer = None
if metadata is not None and config.remote_serde is not None:
    from lmcache.v1.storage_backend.naive_serde import CreateSerde
    self.serializer, self.deserializer = CreateSerde(
        config.remote_serde, metadata, config
    )
    logger.info(f"LocalDiskBackend: Using serializer: {config.remote_serde}")

# VRAM logging
self.vram_log_enabled = os.environ.get("LMCACHE_VRAM_LOG", "0") == "1"
self.vram_log_file = os.environ.get("LMCACHE_VRAM_LOG_FILE", "/tmp/lmcache_vram.log")
```

### 수정 내용 2: async_save_bytes_to_disk에서 압축 적용 (line ~494)
```python
if self.serializer is not None:
    compressed_obj = self.serializer.serialize(memory_obj)
    buffer = compressed_obj.byte_array
    
    if self.vram_log_enabled and torch.cuda.is_available():
        mem_after = torch.cuda.memory_allocated() / (1024**3)
        log_msg = f"[LMCACHE_VRAM][LocalDiskBackend] serialize: +{mem_after - mem_before:.4f}GB (total: {mem_after:.4f}GB)"
        print(log_msg)
        with open(self.vram_log_file, "a") as f:
            f.write(log_msg + "\n")
else:
    buffer = memory_obj.byte_array
```

---

## 5. VRAM 모니터링 로그 예시

```
[LMCACHE_VRAM] 01_split_kv: +0.0000GB (total: 6.1055GB)
[LMCACHE_VRAM] 02_quant_key: +0.0006GB (total: 6.1061GB)
[LMCACHE_VRAM] 03_quant_value: +-0.0024GB (total: 6.1037GB)
[LMCACHE_VRAM] 04_cat_encode_input: +-0.0002GB (total: 6.1035GB)
[LMCACHE_VRAM] 05_calculate_cdf: +0.0001GB (total: 6.1036GB)
[LMCACHE_VRAM] 06_output_buffer: +0.0028GB (total: 6.1065GB)
[LMCACHE_VRAM] 07_encode_ntokens_start: +0.0040GB (total: 6.1104GB)
[LMCACHE_VRAM] 08_encode_ntokens_chunk: total: 6.1105GB
[LMCACHE_VRAM] encode_function: input=6.1055GB, output=6.1105GB, increase=0.0050GB
[LMCACHE_VRAM][LocalDiskBackend] serialize: +0.0019GB (total: 6.1050GB)
```

---

## 6. 실험 실행 방법

### Step 1: 환경 설정

```bash
export LMCACHE_CONFIG_FILE=/home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen.yaml
export LMCACHE_VRAM_LOG=1
export LMCACHE_VRAM_LOG_FILE=/tmp/lmcache_vram.log
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
```

### Step 2: vLLM 시작 (CacheGen 모드)

```bash
/home/noslab-gpu/tkdgjs/tkdgjs/bin/vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --port 8005 \
  --dtype half \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.5 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"discard_partial_chunks":false}}'
```

### Step 3: 테스트 요청 전송

```bash
curl -s http://localhost:8005/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "prompt": "Write a detailed story about a dragon.",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### Step 4: VRAM 측정

```bash
# VRAM 확인
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

# LMCache VRAM 로그 확인
cat /tmp/lmcache_vram.log

# 디스크 캐시 크기 확인
ls -la /home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen_disk/*.pt
```

### Step 5: Native 모드로 재실험

```bash
# 설정 파일만 변경
export LMCACHE_CONFIG_FILE=/home/noslab-gpu/tkdgjs/experiment/lmcache_native.yaml

# vLLM 재시작 후 동일한 실험 수행
```

---

## 7. 그래프 생성 방법

### VRAM 검증 그래프 생성

```bash
/home/noslab-gpu/tkdgjs/tkdgjs/bin/python /home/noslab-gpu/tkdgjs/experiment/generate_vram_verified.py
```

生成된 그래프: `/tmp/vram_verified_breakdown.png`

### 이메일로 전송

```bash
python3 /home/noslab-gpu/tkdgjs/experiment/send_email.py
```

또는 수동:

```python
import email.mime.multipart
import email.mime.image
import smtplib

msg = email.mime.multipart.MIMEMultipart()
msg['Subject'] = '[VRAM Analysis] Results'
msg['From'] = 'your_email@gmail.com'
msg['To'] = 'recipient@example.com'

# Attach image
with open('/tmp/vram_verified_breakdown.png', 'rb') as f:
    img = email.mime.image.MIMEImage(f.read(), 'png')
    img.add_header('Content-Disposition', 'attachment', filename='vram_breakdown.png')
    msg.attach(img)

# Send
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('your_email@gmail.com', 'app_password')
server.send_message(msg)
server.quit()
```

---

## 8. 검증 결과

### nvidia-smi vs Breakdown 비교

| 모드 | nvidia-smi | Breakdown 합계 | 일치율 |
|------|------------|--------------|--------|
| Native | 12.50 GB | 12.50 GB | 100% ✅ |
| CacheGen | 14.05 GB | 14.05 GB | 100% ✅ |

### 결론

- **CacheGen은 Native보다 +1.55 GB 더 많은 VRAM 사용**
- **하지만 7.2x 디스크 공간 절약**
- **VRAM Breakdown이 nvidia-smi 측정값과 완전히 일치함**

---

## 9. 참고 사항

### NVCC 오류가 발생하면

vLLM이 NVCC를 찾지 못해 실패하면 다음 옵션 사용:

```bash
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
```

### GPU 메모리 부족 오류

`--gpu-memory-utilization` 값을 낮추세요:

```bash
--gpu-memory-utilization 0.3  # 30%로 감소
```

### 로그 파일 위치

- vLLM 로그: `/tmp/vllm.log`
- LMCache VRAM 로그: `/tmp/lmcache_vram.log`
- 디스크 캐시: `/home/noslab-gpu/tkdgjs/experiment/lmcache_*_disk/`

---

## 10. 파일 목록

### 생성된 파일

| 파일 | 설명 |
|------|------|
| `lmcache_cachegen.yaml` | CacheGen 설정 |
| `lmcache_native.yaml` | Native 설정 |
| `monitor_vram.py` | VRAM 모니터링 스크립트 |
| `generate_vram_verified.py` | 검증 그래프 생성 |
| `vram_verified_breakdown.png` | 결과 그래프 |

### 수정된 파일

| 파일 | 변경 내용 |
|------|----------|
| `lmcache/v1/storage_backend/local_disk_backend.py` | VRAM 모니터링 및 압축 적용 |

---

*생성일: 2026-02-16*
*실험자: AI Research Team*
