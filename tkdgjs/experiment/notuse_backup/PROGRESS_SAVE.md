# VRAM Experiment Progress - Session Save

## Date
2026-02-15

---

## Goal
Compare VRAM usage between Native and CacheGen modes to prove CacheGen uses more VRAM due to compression buffers.

---

## What Was Done

### 1. LMCache VRAM Logging Code (Modified)
- **File**: `tkdgjs/lib/python3.10/site-packages/lmcache/storage_backend/serde/cachegen_encoder.py`
- **Change**: Added VRAM logging to file
- **Environment Variables**:
  - `LMCACHE_VRAM_LOG=1` - Enable logging
  - `LMCACHE_VRAM_LOG_FILE=/tmp/lmcache_vram.log` - Log file path

### 2. Experiment Script Updated
- **File**: `experiment/run_sweep_experiment.py`
- **Change**: Added LMCACHE_VRAM_LOG environment variables to vLLM startup

### 3. Backup Created
- **Location**: `experiment/lmcache_backup/`
- **Command**: `cp -r tkdgjs/lib/python3.10/site-packages/lmcache/* experiment/lmcache_backup/`

---

## Current Problem

### Issue: Compression Not Triggered

Both Native and CacheGen show same VRAM (10.36 GB) and same disk size (128 MB).

**Logs show:**
```
Native:   remote_serde: torch, size: 0.125 GB
CacheGen: remote_serde: cachegen, size: 0.125 GB
```

**Expected:**
```
Native:   size: ~0.125 GB (no compression)
CacheGen: size: ~0.030 GB (with compression)
```

### Analysis

The `encode_function` (compression function) is NOT being called during the experiment.

Debug prints added to code are NOT showing up in logs, indicating:
1. Compression is skipped/not triggered
2. Or happening in separate process not captured

---

## Files Modified

1. `tkdgjs/lib/python3.10/site-packages/lmcache/storage_backend/serde/cachegen_encoder.py`
   - Added file logging for VRAM measurements
   - Added debug prints

2. `experiment/run_sweep_experiment.py`
   - Added LMCACHE_VRAM_LOG environment variables

---

## Version Management

- **Original**: `experiment/lmcache_backup/`
- **Current**: `tkdgjs/lib/python3.10/site-packages/lmcache/`

### To Restore Original
```bash
cp -r experiment/lmcache_backup/* tkdgjs/lib/python3.10/site-packages/lmcache/
```

---

## Test Results (4096 tokens, gpu_mem=0.7)

| Mode | VRAM | Disk | Latency |
|------|------|------|---------|
| Native | 10.36 GB | 128 MB | 3.71s |
| CacheGen | 10.36 GB | 128 MB | 3.68s |
| **Diff** | **0 GB** | **0 MB** | - |

---

## Next Steps Needed

1. **Verify compression is triggered**
   - Check if CacheGen actually calls `encode_function`
   - May need different LMCache configuration

2. **Alternative approach**
   - Use simulation (measure_vram_breakdown.py) - This worked before
   - Direct measurement with torch.cuda during compression

3. **Check LMCache version**
   - `pip show lmcache` - not found (installed as package)
   - Files at: `tkdgjs/lib/python3.10/site-packages/lmcache/`

---

## Documentation

- `experiment/LMCACHE_VERSION_MANAGEMENT.md` - Version tracking
- `experiment/VRAM_BREAKDOWN.md` - Previous analysis

---

## Git Commits

- Recent commits show VRAM monitoring changes
- Check `git log` for details

---

*This file was created to save progress for another session.*
