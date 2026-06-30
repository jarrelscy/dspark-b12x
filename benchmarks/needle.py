#!/usr/bin/env python3
"""Long-context correctness probe. Embeds a secret code in a long filler context
and asks the model to recall it. If the block-table trim corrupts long-context
attention, retrieval fails / output is garbage."""
import os, json, time, urllib.request
KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = os.environ.get("BENCH_MODEL", "deepseek/v4flash")
BASE = "http://localhost:8001"

def chat(messages, mx=64):
    body = json.dumps({"model": MODEL, "messages": messages,
                       "max_tokens": mx, "temperature": 0.0}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t = time.perf_counter()
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    dt = time.perf_counter() - t
    return d["choices"][0]["message"]["content"], d["usage"], dt

SECRET = "PURPLE-ELEPHANT-7492"
# filler ~ N sentences; each sentence ~ 12 tokens. Target several depths.
def filler(n):
    return " ".join(f"The quarterly report for division {i%50} was filed on schedule." for i in range(n))

for approx_tokens in (4000, 30000, 120000):
    n = approx_tokens // 12
    half = n // 2
    ctx = filler(half) + f"\n\nIMPORTANT: The access code is {SECRET}. Remember it.\n\n" + filler(half)
    msg = [{"role": "user", "content": ctx +
            "\n\nQuestion: What is the access code mentioned above? Reply with only the code."}]
    try:
        out, usage, dt = chat(msg)
        ok = SECRET in out
        print(f"~{approx_tokens:>6} ctx | prompt_toks={usage['prompt_tokens']:>7} | "
              f"{'PASS' if ok else 'FAIL':4} | {dt:5.1f}s | got: {out.strip()[:50]!r}")
    except Exception as e:
        print(f"~{approx_tokens:>6} ctx | ERROR: {e}")
