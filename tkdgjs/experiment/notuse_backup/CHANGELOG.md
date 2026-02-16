# 변경 이력 (Changelog)

## 2026-02-16

### 새로 생성된 파일

| 파일 | 설명 |
|------|------|
| `EXPERIMENT_GUIDE.md` | 종합 실험 가이드 |
| `QUICKSTART.md` | 빠른 시작 가이드 |
| `monitor_vram.py` | VRAM 모니터링 스크립트 |
| `generate_vram_verified.py` | 검증된 VRAM 그래프 생성 스크립트 |
| `generate_vram_layout.py` | VRAM 레이아웃 그래프 생성 |
| `generate_vram_graph_accurate.py` | 정확한 VRAM 그래프 생성 |
| `vram_breakdown.png` | VRAM 분석 그래프 |
| `vram_layout_breakdown.png` | VRAM 레이아웃 그래프 |
| `vram_verified_breakdown.png` | 검증된 VRAM 그래프 |

### 새로 생성된 설정 파일

| 파일 | 설명 |
|------|------|
| `lmcache_cachegen.yaml` | CacheGen 압축 설정 |
| `lmcache_native.yaml` | Native (비압축) 설정 |
| `lmcache_cachegen_remote.yaml` | 리모트 백엔드 설정 (테스트용) |
| `lmcache_torch_remote.yaml` | 리모트 백엔드 설정 (테스트용) |
| `lmcache_cachegen_test.yaml` | 테스트용 설정 |
| `lmcache_torch_test.yaml` | 테스트용 설정 |

### 수정된 파일

#### LMCache 코드 수정

**파일**: `/home/noslab-gpu/tkdgjs/tkdgjs/lib/python3.10/site-packages/lmcache/v1/storage_backend/local_disk_backend.py`

**변경 사항**:

1. **Line ~154**: `__init__`에 serializer 초기화 추가
   - `remote_serde` 설정에 따라 `CacheGenSerializer` 또는 `NaiveSerializer` 선택
   - VRAM 로깅을 위한 환경 변수 (`LMCACHE_VRAM_LOG`, `LMCACHE_VRAM_LOG_FILE`) 처리 추가

2. **Line ~494**: `async_save_bytes_to_disk`에서 압축 적용
   - `serializer.serialize()` 호출로 압축 수행
   - 압축 전후 VRAM 측정 및 로깅

### 발견 및 해결

1. **문제 1**: `local_disk` 백엔드에서 압축이 적용되지 않음
   - **원인**: `remote_serde` 설정이 `remote_url`이 있을 때만 적용됨
   - **해결**: `local_disk_backend.py`에 직접 serializer 적용하는 코드 수정

2. **문제 2**: NVCC 오류로 vLLM 시작 실패
   - **원인**: CUDA toolkit이 설치되어 있지 않음
   - **해결**: `VLLM_ATTENTION_BACKEND=TRITON_ATTN` 환경 변수 사용

3. **문제 3**: GPU 메모리 부족
   - **해결**: `--gpu-memory-utilization` 값을 0.5로 감소

### 실험 결과

| 구분 | Native (Torch) | CacheGen | 차이 |
|------|---------------|----------|------|
| **VRAM (nvidia-smi)** | 12.50 GB | 14.05 GB | +1.55 GB |
| **압축 버퍼** | 0 GB | 1.55 GB | +1.55 GB |
| **디스크 저장** | ~7.9 MB | 1.1 MB | 7.2x 압축 |
| **검증** | ✅ 100% | ✅ 100% | - |

### 이메일 전송

- **수신자**: tkdgjs0213@gmail.com
- **전송 횟수**: 3회
  1. 초기 분석 결과
  2. 정확한 분석 결과  
  3. 검증된 VRAM 레이아웃 그래프

---

## 참고

- 모든 설정 파일과 스크립트는 `/home/noslab-gpu/tkdgjs/experiment/` 디렉토리에 저장됨
- 그래프 이미지는 `/tmp/`와 experiment 폴더에 모두 저장됨
- LMCache VRAM 로그는 `/tmp/lmcache_vram.log`에 저장됨

*Last Updated: 2026-02-16*
