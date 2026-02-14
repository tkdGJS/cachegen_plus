# CacheGen VRAM Experiment

## Overview

This experiment measures and compares VRAM usage between Native (no compression) and CacheGen (compression enabled) modes in LMCache/vLLM inference.

## Hypothesis

CacheGen compression uses **MORE** VRAM than Native mode during compression because it needs:
- Original KV cache
- Compression buffers (quantized tensors, CDF, output buffers)

## Results

### VRAM Usage Comparison (4096 tokens, Llama-3.2-1B)

| Mode | Peak VRAM Increase |
|------|-------------------|
| **CacheGen** (compression) | 1.67 GB |
| **Native** (copy only) | 1.00 GB |
| **Difference** | **+0.67 GB** |

### Conclusion

**Hypothesis CONFIRMED**: CacheGen compression uses 0.67 GB MORE peak VRAM than Native mode.

## Experiment Files

### Test Scripts
- `test_vram_direct.py` - Direct VRAM measurement of CacheGen encoder
- `run_sweep_experiment.py` - Main sweep experiment script
- `quick_test.py` - Quick validation test

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
