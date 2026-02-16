# LMCache VRAM 분석 - 빠른 시작 가이드

## 5분 퀵스타트

### 1. vLLM 시작 (CacheGen 모드)

```bash
# CacheGen 설정
export LMCACHE_CONFIG_FILE=/home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen.yaml
export LMCACHE_VRAM_LOG=1
export LMCACHE_VRAM_LOG_FILE=/tmp/lmcache_vram.log
export VLLM_ATTENTION_BACKEND=TRITON_ATTN

/home/noslab-gpu/tkdgjs/tkdgjs/bin/vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --port 8005 \
  --dtype half \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.5 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"discard_partial_chunks":false}}'
```

### 2. 요청 전송

```bash
curl -s http://localhost:8005/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "prompt": "Write a story.",
    "max_tokens": 100
  }'
```

### 3. VRAM 확인

```bash
# 전체 VRAM
nvidia-smi

# LMCache 로그
cat /tmp/lmcache_vram.log

# 디스크 캐시
ls -la /home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen_disk/
```

### 4. Native 모드로 재실행

```bash
# 설정 파일만 변경
export LMCACHE_CONFIG_FILE=/home/noslab-gpu/tkdgjs/experiment/lmcache_native.yaml

# vLLM 재시작 후 동일한 과정 반복
```

### 5. 그래프 생성

```bash
/home/noslab-gpu/tkdgjs/tkdgjs/bin/python /home/noslab-gpu/tkdgjs/experiment/generate_vram_verified.py
```

생성된 그래프: `/tmp/vram_verified_breakdown.png`

---

## 실험 결과 요약

| 항목 | Native | CacheGen | 차이 |
|------|--------|----------|------|
| VRAM | 12.50 GB | 14.05 GB | +1.55 GB |
| 디스크 | ~7.9 MB | 1.1 MB | 7.2x 압축 |

**결론**: CacheGen은 VRAM을 더 사용하지만 디스크 공간을 7.2배 절약합니다.

---

## 파일 위치

- 설정: `/home/noslab-gpu/tkdgjs/experiment/lmcache_*.yaml`
- 로그: `/tmp/lmcache_vram.log`
- 그래프: `/tmp/vram_verified_breakdown.png`
- 가이드: `/home/noslab-gpu/tkdgjs/experiment/EXPERIMENT_GUIDE.md`
