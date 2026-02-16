#!/usr/bin/env python3
"""
VRAM monitoring using torch.cuda during LLM inference
Tracks torch CUDA memory to detect temporary VRAM spikes
"""
import sys
import time
import json
import subprocess
import threading

# Add vLLM environment
sys.path.insert(0, '/home/noslab-gpu/tkdgjs/tkdgjs/lib/python3.10/site-packages')

VLLM_PORT = 8000

class TorchVRAMMonitor:
    def __init__(self, interval=0.01):
        self.interval = interval
        self.running = False
        self.samples = []
        self.start_time = None
        self.thread = None
        
    def _monitor_loop(self):
        import torch
        while self.running:
            try:
                allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
                max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
                
                elapsed = time.time() - self.start_time
                
                self.samples.append({
                    'elapsed_sec': elapsed,
                    'timestamp': time.time(),
                    'torch_allocated_gb': allocated,
                    'torch_reserved_gb': reserved,
                    'torch_peak_gb': max_allocated,
                })
            except Exception as e:
                pass
            
            time.sleep(self.interval)
    
    def start(self):
        import torch
        torch.cuda.reset_peak_memory_stats()
        self.running = True
        self.start_time = time.time()
        self.samples = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def get_samples(self):
        return self.samples


def get_nvidia_smi_vram():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', '0'],
        capture_output=True, text=True, timeout=5
    )
    return float(result.stdout.strip())


def send_request(prompt_tokens, max_tokens=32):
    import urllib.request
    import urllib.error
    
    prompt = "word " * prompt_tokens
    data = json.dumps({
        'model': 'meta-llama/Llama-3.2-1B-Instruct',
        'prompt': prompt,
        'max_tokens': max_tokens,
        'temperature': 0.0
    }).encode()
    
    req = urllib.request.Request(
        f'http://localhost:{VLLM_PORT}/v1/completions',
        data=data,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = response.read()
            elapsed = time.time() - start
            return True, elapsed
    except Exception as e:
        return False, str(e)


def run_experiment(mode, prefill_size):
    print(f"\n=== {mode}, {prefill_size} tokens ===")
    
    # Start vLLM
    print("Starting vLLM...")
    proc = start_vllm(mode)
    if not proc:
        return None
    
    time.sleep(40)  # Warmup
    
    # Start monitoring
    monitor = TorchVRAMMonitor(interval=0.01)
    monitor.start()
    
    time.sleep(2)  # Baseline
    
    # Send request
    print(f"Sending {prefill_size} token request...")
    success, result = send_request(prefill_size)
    
    time.sleep(5)  # Wait for cleanup
    
    # Stop monitoring
    monitor.stop()
    samples = monitor.get_samples()
    
    # Get final VRAM
    final_vram = get_nvidia_smi_vram()
    
    # Stop vLLM
    stop_vllm()
    
    # Analyze results
    if samples:
        allocated = [s['torch_allocated_gb'] for s in samples]
        reserved = [s['torch_reserved_gb'] for s in samples]
        
        print(f"Samples: {len(samples)}")
        print(f"torch_allocated: min={min(allocated):.4f}, max={max(allocated):.4f} GB")
        print(f"torch_reserved: min={min(reserved):.4f}, max={max(reserved):.4f} GB")
        print(f"nvidia-smi VRAM: {final_vram:.0f} MB")
    
    return {
        'mode': mode,
        'prefill_size': prefill_size,
        'samples': samples,
        'final_vram_mb': final_vram,
        'success': success,
    }


def start_vllm(mode):
    import subprocess
    
    config_path = f'/home/noslab-gpu/tkdgjs/experiment/lmcache_{mode}.yaml'
    disk_path = f'/home/noslab-gpu/tkdgjs/experiment/lmcache_{mode}_disk'
    
    subprocess.run(['pkill', '-f', 'vllm'], stderr=subprocess.DEVNULL)
    time.sleep(3)
    
    cmd = [
        '/home/noslab-gpu/tkdgjs/tkdgjs/bin/vllm', 'serve', 'meta-llama/Llama-3.2-1B-Instruct',
        '--port', str(VLLM_PORT),
        '--dtype', 'half',
        '--max-model-len', '8192',
        '--gpu-memory-utilization', '0.7',
        '--disable-hybrid-kv-cache-manager',
        '--kv-transfer-config', '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"discard_partial_chunks":false}}',
        '--enforce-eager',
        '--attention-backend', 'triton_attn',
    ]
    
    env = {
        'LMCACHE_CONFIG_FILE': config_path,
        'PATH': '/home/noslab-gpu/tkdgjs/tkdgjs/bin:/home/noslab-gpu/.pyenv/shims:/usr/local/bin:/usr/bin:/bin',
    }
    
    log_file = f'/home/noslab-gpu/tkdgjs/experiment/vllm_torch_{mode}.log'
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=f)
    
    # Wait for vLLM to be ready
    import requests
    for _ in range(90):
        try:
            resp = requests.get(f'http://localhost:{VLLM_PORT}/v1/models', timeout=5)
            if resp.status_code == 200:
                return proc
        except:
            pass
        time.sleep(2)
    
    return None


def stop_vllm():
    import subprocess
    subprocess.run(['pkill', '-f', 'vllm'], stderr=subprocess.DEVNULL)
    time.sleep(3)


if __name__ == '__main__':
    results = {}
    
    configs = [
        ('native', 4096),
        ('cachegen', 4096),
    ]
    
    for mode, prefill in configs:
        result = run_experiment(mode, prefill)
        if result:
            results[f'{mode}_{prefill}'] = result
            
            # Save timeseries
            filename = f'/home/noslab-gpu/tkdgjs/experiment/vram_torch_{mode}_p{prefill}.jsonl'
            with open(filename, 'w') as f:
                for sample in result['samples']:
                    f.write(json.dumps(sample) + '\n')
            print(f"Saved to {filename}")
    
    print("\n=== Summary ===")
    for key, data in results.items():
        if data['samples']:
            allocated = [s['torch_allocated_gb'] for s in data['samples']]
            print(f"{key}: torch_alloc peak={max(allocated):.4f} GB, nvidia-smi={data['final_vram_mb']:.0f} MB")
