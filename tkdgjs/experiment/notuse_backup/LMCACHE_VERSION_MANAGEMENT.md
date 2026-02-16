# LMCache Version Management & Modification Log

## Overview

This document tracks LMCache source code modifications for VRAM measurement during compression.

---

## Version History

### v1.0 - Original (2026-02-15)
- **Status**: Original (before modification)
- **Location**: `tkdgjs/lib/python3.10/site-packages/lmcache/`
- **Backup**: `experiment/lmcache_backup/`
- **Description**: Original LMCache source code

### v1.1 - Modified (TBD)
- **Status**: Planned
- **Modification**: Add VRAM measurement before compression buffer free
- **Target Files**: 
  - `lmcache/v1/gpu_connector.py`
  - `lmcache/storage_backend/serde/cachegen_encoder.py`

---

## Backup Information

### Backup Date
2026-02-15

### Backup Command
```bash
cp -r /home/noslab-gpu/tkdgjs/tkdgjs/lib/python3.10/site-packages/lmcache/* /home/noslab-gpu/tkdgjs/experiment/lmcache_backup/
```

### Backup Contents
```
experiment/lmcache_backup/
├── config.py
├── connections.py
├── c_ops.cpython-310-x86_64-linux-gnu.so
├── __init__.py
├── integration/
├── logging.py
├── native_storage_ops.cpython-310-x86_64-linux-gnu.so
├── storage_backend/
├── usage_context.py
├── utils.py
├── v1/
└── ... (all LMCache files)
```

---

## Modification Plan

### Objective
Measure VRAM usage immediately **before** compression buffer is freed to capture CacheGen's additional memory usage.

### Target Code Location

**File**: `lmcache/v1/gpu_connector.py`
**Line**: ~871 (before `ref_count_down()`)

```python
# Current code:
# free the buffer memory
load_gpu_buffer_obj.ref_count_down()
compute_gpu_buffer_obj.ref_count_down()

# Modified code:
# free the buffer memory
vram_before_free = torch.cuda.memory_allocated() / (1024**3)
logger.info(f"LMCACHE_VRAM: Before free = {vram_before_free:.4f} GB")
load_gpu_buffer_obj.ref_count_down()
compute_gpu_buffer_obj.ref_count_down()
```

### Alternative Location

**File**: `lmcache/storage_backend/serde/cachegen_encoder.py`

---

## Restoration Commands

### Restore to Original
```bash
# Method 1: Copy from backup
cp -r /home/noslab-gpu/tkdgjs/experiment/lmcache_backup/* /home/noslab-gpu/tkdgjs/tkdgjs/lib/python3.10/site-packages/lmcache/

# Method 2: Reinstall
pip install lmcache --force-reinstall
```

### Verify Restoration
```bash
# Check if files match backup
diff -r /home/noslab-gpu/tkdgjs/experiment/lmcache_backup/ /home/noslab-gpu/tkdgjs/tkdgjs/lib/python3.10/site-packages/lmcache/
```

---

## Testing Procedure

### 1. Before Modification
- Run baseline experiment with native and cachegen
- Record VRAM measurements

### 2. After Modification
- Run same experiment
- Check logs for `LMCACHE_VRAM` messages
- Compare results

### 3. Expected Results

| Mode | Before Free VRAM | Notes |
|------|------------------|-------|
| Native | ~10.36 GB | No compression buffers |
| CacheGen | ~10.36+ GB | Compression buffers should show additional VRAM |

---

## Notes

- LMCache is installed via pip, not git-managed
- This modification is for **measurement only**, not production use
- After experiments, restore original code

---

## Contact

For questions about this modification, refer to the VRAM experiment documentation.

---

*Last updated: 2026-02-15*
