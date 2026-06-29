#!/usr/bin/env python3
"""Broad single-stream decode benchmark. Set MODEL to deepseek/v4flashdspark or deepseek/v4flash."""
import os, time, json, urllib.request
KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = os.environ.get("BENCH_MODEL", "deepseek/v4flashdspark")
URL = "http://localhost:8001/v1/chat/completions"
def gen(p, mx=256):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": p}],
                       "max_tokens": mx, "temperature": 0.0}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t = time.perf_counter()
    d = json.loads(urllib.request.urlopen(req, timeout=120).read())
    return d["usage"]["completion_tokens"] / (time.perf_counter() - t)
prompts = ["Write a 200-word essay on the Roman Empire.", "Explain how photosynthesis works.",
           "Write a short story about a dragon.", "Summarize the plot of Hamlet.",
           "Explain how a neural network learns.", "Describe the water cycle.",
           "Write a Python function to sort a list.", "Explain quantum entanglement simply."]
for p in prompts[:2]: gen(p, 64)  # warm
sp = [gen(p) for p in prompts]
print("per-prompt tok/s:", " ".join(f"{x:.0f}" for x in sp))
print(f"AVG: {sum(sp)/len(sp):.1f} tok/s  (min {min(sp):.0f}, max {max(sp):.0f})")
