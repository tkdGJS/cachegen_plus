# CacheGen VRAM Experiment

## Overview

This experiment measures and compares VRAM usage between Native (no compression) and CacheGen (compression enabled) modes in LMCache/vLLM inference.

## Hypothesis

CacheGen compression uses **MORE** VRAM than Native mode during compression because it needs:
- Original KV cache
- Compression buffers (quantized tensors, CDF, output buffers)

## Results

### Full Sweep Experiment (2026-02-14)

**Configuration:**
- GPU Util: 0.3, 0.5, 0.7, 0.9
- Prefill Sizes: 128, 256, 512, 1024, 2048 tokens
- Modes: native, cachegen
- Model: meta-llama/Llama-3.2-1B-Instruct
- Total Experiments: 40 (all successful, 0 OOM)

**Summary:**
| Metric                  | Native  | CacheGen |
|------------------------|---------|----------|
| Avg Latency (s)        | 0.62    | 0.62     |
| Avg Disk Offload (MB)  | 0.00    | 39.61    |
| Avg Compression Ratio  | 0.0262  | 0.0262   |

**OOM Events:** None (all 40 experiments completed successfully)

### VRAM Usage by GPU Utilization

| GPU Util | VRAM After Start (GB) |
|----------|----------------------|
| 0.3      | 4.84                 |
| 0.5      | 7.77                 |
| 0.7      | 10.68                |
| 0.9      | 13.59                |

### VRAM Usage Comparison (Direct Test - 4096 tokens)

| Mode | Peak VRAM Increase |
|------|-------------------|
| **CacheGen** (compression) | 1.67 GB |
| **Native** (copy only) | 1.00 GB |
| **Difference** | **+0.67 GB** |

### Conclusion

**Hypothesis CONFIRMED**: CacheGen compression uses 0.67 GB MORE peak VRAM than Native mode during encoding operation.

## Experiment Files

### Test Scripts
- `test_vram_direct.py` - Direct VRAM measurement of CacheGen encoder
- `run_full_sweep.py` - Main sweep experiment script (gpu_util 0.3-0.9, prefill 128-2048)
- `quick_test.py` - Quick validation test

### Output Files
- `sweep_results_latest.json` - Latest sweep results
- `backup_20260214/` - Backup of all results
- `timeline_*.jsonl` - Individual experiment timeline
- `vram_timeline.log` - VRAM timeline log
- `oom_events.log` - OOM event log (empty - no OOM occurred)

### LMCache Config Files
- `lmcache_cachegen.yaml` - CacheGen mode (compression enabled)
- `lmcache_native.yaml` - Native mode (no compression)

## Measurement Method

### Method Used: torch.cuda.memory_allocated()

1. Added VRAM measurement to LMCache `encode_function`:
   ```python
   mem_before = torch.cuda.memory_allocated()
   # ... compression ...
   mem_after = torch.cuda.memory_allocated()
   ```

2. Direct test with 1GB KV cache (4096 tokens):
   - CacheGen: encodes KV cache with compression buffers
   - Native: copies KV cache without compression

## Modified LMCache Code

File: `tkdgjs/tkdgjs/lib/python3.10/site-packages/lmcache/storage_backend/serde/cachegen_encoder.py`

Added VRAM measurement before/after `encode_function`:
```python
mem_before = torch.cuda.memory_allocated() / (1024**3)
# ... compression ...
mem_after = torch.cuda.memory_allocated() / (1024**3)
mem_increase = mem_after - mem_before

if os.environ.get("LMCACHE_VRAM_LOG", "0") == "1":
    print(f"[LMCACHE_VRAM] encode_function: before={mem_before:.4f}GB, after={mem_after:.4f}GB, increase={mem_increase:.4f}GB")
```

## Running the Experiment

### Quick Test
```bash
cd /home/noslab-gpu/tkdgjs/experiment
python3 test_vram_direct.py
```

### Full Sweep
```bash
python3 run_sweep_experiment.py
```

## VRAM Layout Monitoring

The experiment tracks 5 VRAM regions:
1. Model Weights (~2GB)
2. KV Cache Allocated
3. Activation Tensors
4. CUDA Runtime
5. CacheGen Buffers (compression)

## Notes

- VRAM spike during compression is transient (~ms)
- nvidia-smi (50ms interval) cannot detect the spike
- torch.cuda.memory_allocated() is required for accurate measurement
- Peak VRAM increase is the key metric, not steady-state VRAM
