# LMCache VRAM 모니터링 설정 문서

## 개요

이 문서는 LMCache의 VRAM 모니터링을 설정하는 방법을 설명합니다.

## 모니터링 코드 적용 상태

### VRAM 모니터링 코드 위치

**수정된 파일** (VRAM 모니터링 포함):
```
/home/noslab-gpu/tkdgjs/tkdgjs/lib/python3.10/site-packages/lmcache/storage_backend/serde/cachegen_encoder.py
```

**실제 사용 경로** (수정된 파일을 import):
```
/home/noslab-gpu/tkdgjs/tkdgjs/lib/python3.10/site-packages/lmcache/v1/storage_backend/naive_serde/cachegen_encoder.py
```

**코드 연결 구조**:
```python
# v1/cachegen_encoder.py (실제 사용)
from lmcache.storage_backend.serde.cachegen_encoder import encode_function  # ← 수정된 파일을 import
```

→ **수정된 코드가 실제로 사용됩니다!**

## 환경 변수 설정

VRAM 모니터링을 활성화하려면 다음 환경 변수를 설정하세요:

```bash
export LMCACHE_CONFIG_FILE=/home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen.yaml
export LMCACHE_VRAM_LOG=1
export LMCACHE_VRAM_LOG_FILE=/tmp/lmcache_vram.log
```

## vLLM 실행 명령어

```bash
/home/noslab-gpu/tkdgjs/tkdgjs/bin/vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --port 8000 \
  --dtype half \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.7 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"discard_partial_chunks":false}}'
```

## LMCache 설정 파일

### CacheGen 모드 (압축)
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

### Native/Torch 모드 (압축 없음)
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

## 발견된 문제점

### 1. 압축이 실제로 발생하지 않음

**현상**: CacheGen과 Torch 모드 모두 디스크 파일 크기가 동일 (8MB)

**근거**:
- VRAM 로그에 `encode_function called` 메시지 없음 → 압축 함수 미호출
- 디스크 파일 크기가 동일 → 압축 데이터가 아님

**원인**: `local_disk` 설정만 있고 `remote_url`이 없음!

```python
# storage_backend/__init__.py 분석

# local_disk 사용시 → LocalDiskBackend → 압축 없음
if config.local_disk and config.max_local_disk_size > 0:
    local_disk_backend = LocalDiskBackend(...)  # 원본 저장

# remote_url 사용시 → RemoteBackend → remote_serde 적용
if config.remote_url is not None:
    remote_backend = RemoteBackend(...)  # 압축 사용!
```

| 현재 설정 | 백엔드 | 압축 |
|-----------|--------|------|
| `local_disk: file:///...` | LocalDiskBackend | ❌ |
| `remote_url: (없음)` | - | - |

**결론**: `remote_serde: cachegen` 설정은 `remote_url`이 있을 때만 적용됨. 로컬 디스크 저장소에는 압축이 적용되지 않음!

### 2. LMCache 아키텍처

```
vLLM → LMCacheConnectorV1 → LMCacheEngine 
                                      ↓
                              StorageManager
                                      ↓
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
            LocalCPUBackend    LocalDiskBackend   RemoteBackend
                    ↓                 ↓                 ↓
            (CPU 메모리)      (디스크 저장)       (원격 저장소)
                                      ↓
                              serializer.serialize()
                                      ↓
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
            NaiveSerializer    CacheGenSerializer
            (원본 저장)         (압축 저장)
```

### 3. LocalDiskBackend는 압축을 사용하지 않음

`local_disk_backend.py`의 `async_save_bytes_to_disk` 함수는:
- `memory_obj.byte_array`를 직접 디스크에 저장
- 압축 없이 원본 데이터 저장
- `remote_serde` 설정과 무관하게 동작

## VRAM 로그 출력 예시

모니터링이 정상 작동하면 다음과 같은 로그가 출력됩니다:

```
[LMCACHE_VRAM] 01_split_kv: +0.0000GB (total: 10.5000GB)
[LMCACHE_VRAM] 02_quant_key: +0.1000GB (total: 10.6000GB)
[LMCACHE_VRAM] 03_quant_value: +0.1000GB (total: 10.7000GB)
[LMCACHE_VRAM] 04_cat_encode_input: +0.0500GB (total: 10.7500GB)
[LMCACHE_VRAM] 05_calculate_cdf: +0.0200GB (total: 10.7700GB)
[LMCACHE_VRAM] 06_output_buffer: +0.1000GB (total: 10.8700GB)
[LMCACHE_VRAM] encode_function: input=10.5000GB, output=10.9500GB, increase=0.4500GB
```

## 테스트된 파일 위치

- 설정 파일: `/home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen.yaml`
- 설정 파일: `/home/noslab-gpu/tkdgjs/experiment/lmcache_native.yaml`
- 디스크 캐시: `/home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen_disk/`
- 디스크 캐시: `/home/noslab-gpu/tkdgjs/experiment/lmcache_torch_disk/`
- VRAM 로그: `/tmp/lmcache_vram.log`

## 다음 단계

1. **압축 활성화 확인**: LMCache + vLLM 통합에서 압축이 실제로 발생하도록 설정 필요
2. **LocalDiskBackend 수정**: 압축을 적용하려면 `local_disk_backend.py`에서 직접 압축 로직 추가 필요
3. **Alternative**: RemoteBackend 사용 - 원격 저장소 연동 시 압축이 발생할 수 있음
