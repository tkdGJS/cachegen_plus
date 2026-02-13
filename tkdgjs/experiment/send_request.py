#!/usr/bin/env python3
"""
Send request to vLLM and measure VRAM during request processing
"""
import sys
import os
import time
import requests
import json

def send_request(prompt_size, port=8000, model="meta-llama/Llama-3.2-1B-Instruct", max_tokens=32):
    """Send completion request to vLLM"""
    
    # Create prompt of exact token size
    # Using repeating characters to control token count approximately
    # Llama3 uses ~4 chars per token, so we multiply
    prompt = "A " * (prompt_size * 2)  # Approximate token count
    
    url = f"http://localhost:{port}/v1/completions"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0
    }
    
    print(f"[Request] Sending request with prompt_size={prompt_size}")
    print(f"[Request] URL: {url}")
    print(f"[Request] Payload: prompt length ~{len(prompt)} chars, max_tokens={max_tokens}")
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"[Request] Success! Elapsed: {elapsed:.2f}s")
            print(f"[Request] Response: {json.dumps(result, indent=2)[:500]}")
            return True, elapsed, result
        else:
            print(f"[Request] Error: {response.status_code}")
            print(f"[Request] Response: {response.text}")
            return False, elapsed, None
            
    except Exception as e:
        print(f"[Request] Exception: {e}")
        return False, 0, None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-size", type=int, default=256, help="Approximate prompt size in tokens")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    
    success, elapsed, result = send_request(args.prompt_size, args.port, max_tokens=args.max_tokens)
    sys.exit(0 if success else 1)
