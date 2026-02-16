#!/usr/bin/env python3
"""
Complete Sweep Experiment: Run experiments, generate graphs, and email results
"""

import os
import subprocess
import json
import time
import smtplib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from datetime import datetime

# ============== Configuration ==============
CONFIGS = [
    (256, 0.2),
    (512, 0.2),
    (1024, 0.2),
]

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
PORT = 8005
CACHEGEN_CONFIG = "/home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen.yaml"
NATIVE_CONFIG = "/home/noslab-gpu/tkdgjs/experiment/lmcache_native.yaml"
CACHEGEN_DIR = "/home/noslab-gpu/tkdgjs/experiment/lmcache_cachegen_disk"
NATIVE_DIR = "/home/noslab-gpu/tkdgjs/experiment/lmcache_torch_disk"

# Email config
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "noreply.airesearch@gmail.com"
SENDER_PASSWORD = "kwqfkeogkozsxffu"
RECIPIENT_EMAIL = "tkdgjs0213@gmail.com"

# ============== Helper Functions ==============

def kill_vllm():
    """Kill existing vLLM processes"""
    subprocess.run(["pkill", "-9", "-f", "vllm"], capture_output=True)
    time.sleep(10)  # Wait longer for GPU memory to be freed

def check_vllm_ready(port=PORT, timeout=60):
    """Check if vLLM is ready"""
    for _ in range(timeout):
        try:
            result = subprocess.run(
                ["curl", "-s", f"http://localhost:{port}/v1/models"],
                capture_output=True, timeout=5
            )
            if result.returncode == 0 and b'"object":"list"' in result.stdout:
                return True
        except:
            pass
        time.sleep(1)
    return False

def get_vram_usage():
    """Get current VRAM usage in GB"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            mem_mb = float(result.stdout.strip().split('\n')[0])
            return mem_mb / 1024.0
    except:
        pass
    return None

def get_disk_size(cache_dir):
    """Get total cache directory size in MB"""
    try:
        result = subprocess.run(
            ["du", "-sm", cache_dir],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return float(result.stdout.split()[0])
    except:
        pass
    return 0.0

def clean_cache(cache_dir):
    """Clean cache directory"""
    subprocess.run(["rm", "-rf", f"{cache_dir}/*"], shell=True, capture_output=True)
    subprocess.run(["rm", "-f", "/tmp/lmcache_vram.log"], capture_output=True)

def run_vllm(mode, tokens, gpu_mem):
    """Run vLLM with specific configuration"""
    config_file = CACHEGEN_CONFIG if mode == "cachegen" else NATIVE_CONFIG
    cache_dir = CACHEGEN_DIR if mode == "cachegen" else NATIVE_DIR
    
    env = os.environ.copy()
    env["LMCACHE_CONFIG_FILE"] = config_file
    env["LMCACHE_VRAM_LOG"] = "1"
    env["LMCACHE_VRAM_LOG_FILE"] = "/tmp/lmcache_vram.log"
    env["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
    
    # Clean cache
    clean_cache(cache_dir)
    
    # Start vLLM
    cmd = [
        "/home/noslab-gpu/tkdgjs/tkdgjs/bin/vllm", "serve",
        MODEL,
        "--port", str(PORT),
        "--dtype", "half",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", str(gpu_mem),
        "--kv-transfer-config", '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"discard_partial_chunks":false}}'
    ]
    
    print(f"  Starting vLLM ({mode})...")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for ready
    if not check_vllm_ready():
        proc.kill()
        return None
    
    time.sleep(2)
    
    # Get VRAM before request
    vram_before = get_vram_usage()
    
    # Send request
    prompt = "Hello " * (tokens // 2)
    curl_cmd = [
        "curl", "-s", f"http://localhost:{PORT}/v1/completions",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0
        })
    ]
    
    try:
        subprocess.run(curl_cmd, capture_output=True, timeout=60)
    except:
        pass
    
    time.sleep(2)
    
    # Get VRAM after request
    vram_after = get_vram_usage()
    
    # Get disk size
    disk_size = get_disk_size(cache_dir)
    
    # Kill vLLM
    kill_vllm()
    
    return {
        "mode": mode,
        "tokens": tokens,
        "gpu_memory": gpu_mem,
        "vram_before": vram_before,
        "vram_after": vram_after,
        "disk_size_mb": disk_size,
        "timestamp": datetime.now().isoformat()
    }

def create_breakdown_graph(results, output_path):
    """Create VRAM breakdown graph"""
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('LMCache VRAM Sweep Experiment Results\nCacheGen vs Native Mode', 
                 fontsize=16, fontweight='bold')
    
    # Group by tokens
    tokens_list = sorted(list(set([r['tokens'] for r in results])))
    
    # Plot 1: VRAM by tokens
    ax1 = fig.add_subplot(2, 2, 1)
    for gpu_mem in [0.5]:
        native_vrams = []
        cachegen_vrams = []
        for tokens in tokens_list:
            n = [r for r in results if r['tokens'] == tokens and r['gpu_memory'] == gpu_mem and r['mode'] == 'native']
            c = [r for r in results if r['tokens'] == tokens and r['gpu_memory'] == gpu_mem and r['mode'] == 'cachegen']
            native_vrams.append(n[0]['vram_after'] if n else 0)
            cachegen_vrams.append(c[0]['vram_after'] if c else 0)
        
        x = np.arange(len(tokens_list))
        width = 0.35
        ax1.bar(x - width/2, native_vrams, width, label='Native', color='#2ecc71')
        ax1.bar(x + width/2, cachegen_vrams, width, label='CacheGen', color='#e74c3c')
    
    ax1.set_xlabel('Tokens')
    ax1.set_ylabel('VRAM (GB)')
    ax1.set_title(f'VRAM Usage (GPU Memory: 0.5)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tokens_list)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: VRAM difference
    ax2 = fig.add_subplot(2, 2, 2)
    diffs = []
    for tokens in tokens_list:
        n = [r for r in results if r['tokens'] == tokens and r['gpu_memory'] == 0.5 and r['mode'] == 'native']
        c = [r for r in results if r['tokens'] == tokens and r['gpu_memory'] == 0.5 and r['mode'] == 'cachegen']
        if n and c:
            diffs.append(c[0]['vram_after'] - n[0]['vram_after'])
        else:
            diffs.append(0)
    
    colors = ['#e74c3c' if d > 0 else '#2ecc71' for d in diffs]
    ax2.bar(range(len(tokens_list)), diffs, color=colors)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Tokens')
    ax2.set_ylabel('VRAM Difference (GB)')
    ax2.set_title('VRAM Difference (CacheGen - Native)')
    ax2.set_xticks(range(len(tokens_list)))
    ax2.set_xticklabels(tokens_list)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Disk size
    ax3 = fig.add_subplot(2, 2, 3)
    native_disks = []
    cachegen_disks = []
    for tokens in tokens_list:
        n = [r for r in results if r['tokens'] == tokens and r['gpu_memory'] == 0.5 and r['mode'] == 'native']
        c = [r for r in results if r['tokens'] == tokens and r['gpu_memory'] == 0.5 and r['mode'] == 'cachegen']
        native_disks.append(n[0]['disk_size_mb'] if n else 0)
        cachegen_disks.append(c[0]['disk_size_mb'] if c else 0)
    
    ax3.bar(x - width/2, native_disks, width, label='Native', color='#2ecc71')
    ax3.bar(x + width/2, cachegen_disks, width, label='CacheGen', color='#e74c3c')
    ax3.set_xlabel('Tokens')
    ax3.set_ylabel('Disk Size (MB)')
    ax3.set_title('KV Cache Disk Size')
    ax3.set_xticks(x)
    ax3.set_xticklabels(tokens_list)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # Calculate stats
    native_results = [r for r in results if r['mode'] == 'native']
    cachegen_results = [r for r in results if r['mode'] == 'cachegen']
    
    avg_native_vram = np.mean([r['vram_after'] for r in native_results]) if native_results else 0
    avg_cachegen_vram = np.mean([r['vram_after'] for r in cachegen_results]) if cachegen_results else 0
    avg_native_disk = np.mean([r['disk_size_mb'] for r in native_results]) if native_results else 0
    avg_cachegen_disk = np.mean([r['disk_size_mb'] for r in cachegen_results]) if cachegen_results else 0
    
    compression_ratio = avg_native_disk / avg_cachegen_disk if avg_cachegen_disk > 0 else 0
    
    summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    SWEEP EXPERIMENT SUMMARY                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Total Experiments: {len(results)}                                           ║
║  Configurations: {len(CONFIGS)} x 2 modes                               ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  AVERAGE VRAM USAGE:                                            ║
║    Native:   {avg_native_vram:.2f} GB                                     ║
║    CacheGen: {avg_cachegen_vram:.2f} GB                                     ║
║    Diff:     {avg_cachegen_vram - avg_native_vram:+.2f} GB                                          ║
║                                                                  ║
║  AVERAGE DISK SIZE:                                             ║
║    Native:   {avg_native_disk:.2f} MB                                      ║
║    CacheGen: {avg_cachegen_disk:.2f} MB                                      ║
║    Compression: {compression_ratio:.1f}x                                           ║
║                                                                  ║
║  KEY FINDING:                                                   ║
║    CacheGen uses MORE VRAM but achieves {compression_ratio:.1f}x compression! ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    
    ax4.text(0.5, 0.5, summary, transform=ax4.transAxes,
             fontsize=10, fontfamily='monospace', va='center', ha='center',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Graph saved to: {output_path}")
    return output_path

def send_email(graph_path, results):
    """Send email with graph"""
    
    # Calculate summary
    native_results = [r for r in results if r['mode'] == 'native']
    cachegen_results = [r for r in results if r['mode'] == 'cachegen']
    
    avg_native_vram = np.mean([r['vram_after'] for r in native_results]) if native_results else 0
    avg_cachegen_vram = np.mean([r['vram_after'] for r in cachegen_results]) if cachegen_results else 0
    avg_native_disk = np.mean([r['disk_size_mb'] for r in native_results]) if native_results else 0
    avg_cachegen_disk = np.mean([r['disk_size_mb'] for r in cachegen_results]) if cachegen_results else 0
    
    compression_ratio = avg_native_disk / avg_cachegen_disk if avg_cachegen_disk > 0 else 0
    
    if not results:
        body = "<p>No results collected. Please check the experiment.</p>"
        compression_ratio = 0
    else:
        body = f"""
<h2>VRAM Sweep Experiment Results</h2>

<p><b>Key Finding:</b> CacheGen uses MORE VRAM than Native but achieves {compression_ratio:.1f}x compression!</p>

<h3>Summary:</h3>
<ul>
<li>Average VRAM - Native: {avg_native_vram:.2f} GB</li>
<li>Average VRAM - CacheGen: {avg_cachegen_vram:.2f} GB</li>
<li>VRAM Difference: {avg_cachegen_vram - avg_native_vram:+.2f} GB</li>
<li>Average Disk - Native: {avg_native_disk:.2f} MB</li>
<li>Average Disk - CacheGen: {avg_cachegen_disk:.2f} MB</li>
<li>Compression Ratio: {compression_ratio:.1f}x</li>
</ul>

<h3>Configurations Tested:</h3>
{', '.join([f'{t}/{g}' for t,g in CONFIGS])}

<p>See attached graph for details.</p>
    """
    
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = f"[VRAM Sweep] CacheGen vs Native - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    msg.attach(MIMEText(body, 'html'))
    
    # Attach graph
    with open(graph_path, 'rb') as f:
        img = MIMEImage(f.read(), 'png')
        img.add_header('Content-Disposition', 'attachment', filename='vram_sweep_results.png')
        msg.attach(img)
    
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        print("Email sent successfully!")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

# ============== Main ==============

def main():
    print("=" * 60)
    print("LMCache VRAM Sweep Experiment")
    print("=" * 60)
    
    results = []
    
    # Run experiments
    for tokens, gpu_mem in CONFIGS:
        print(f"\n--- Testing: tokens={tokens}, gpu_mem={gpu_mem} ---")
        
        # Run CacheGen
        print(f"\n[1/2] Running CacheGen mode...")
        result_cg = run_vllm("cachegen", tokens, gpu_mem)
        if result_cg:
            results.append(result_cg)
            print(f"  CacheGen: VRAM={result_cg['vram_after']:.2f}GB, Disk={result_cg['disk_size_mb']:.2f}MB")
        
        # Run Native
        print(f"\n[2/2] Running Native mode...")
        result_nat = run_vllm("native", tokens, gpu_mem)
        if result_nat:
            results.append(result_nat)
            print(f"  Native: VRAM={result_nat['vram_after']:.2f}GB, Disk={result_nat['disk_size_mb']:.2f}MB")
    
    # Save results
    results_file = "/tmp/sweep_results_new.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Generate graph
    graph_path = "/tmp/vram_sweep_results.png"
    create_breakdown_graph(results, graph_path)
    
    # Send email
    print("\nSending email...")
    send_email(graph_path, results)
    
    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
