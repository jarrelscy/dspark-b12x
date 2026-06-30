#!/usr/bin/env python3
"""Per-prompt DSpark acceptance + tok/s via /metrics deltas around each prompt."""
import os, time, json, re, urllib.request
KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = os.environ.get("BENCH_MODEL", "deepseek/v4flash")
BASE = "http://localhost:8001"

def metrics():
    req = urllib.request.Request(BASE + "/metrics", headers={"Authorization": f"Bearer {KEY}"})
    txt = urllib.request.urlopen(req, timeout=30).read().decode()
    out = {"per_pos": {}, "drafts": 0.0, "accepted": 0.0}
    for ln in txt.splitlines():
        if ln.startswith("#"):
            continue
        if "spec_decode_num_accepted_tokens_per_pos_total" in ln:
            m = re.search(r'position="(\d+)".*?\s([\d.eE+]+)$', ln)
            if m: out["per_pos"][int(m.group(1))] = float(m.group(2))
        elif "spec_decode_num_drafts_total" in ln:
            m = re.search(r"\s([\d.eE+]+)$", ln)
            if m: out["drafts"] = float(m.group(1))
        elif "spec_decode_num_accepted_tokens_total" in ln:
            m = re.search(r"\s([\d.eE+]+)$", ln)
            if m: out["accepted"] = float(m.group(1))
    return out

def gen(p, mx=256):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": p}],
                       "max_tokens": mx, "temperature": 0.0}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t = time.perf_counter()
    d = json.loads(urllib.request.urlopen(req, timeout=180).read())
    return d["usage"]["completion_tokens"] / (time.perf_counter() - t)

prompts = [("essay", "Write a 200-word essay on the Roman Empire."),
           ("photosynth", "Explain how photosynthesis works."),
           ("dragon", "Write a short story about a dragon."),
           ("hamlet", "Summarize the plot of Hamlet."),
           ("neuralnet", "Explain how a neural network learns."),
           ("watercycle", "Describe the water cycle."),
           ("pysort", "Write a Python function to sort a list."),
           ("quantum", "Explain quantum entanglement simply.")]
gen(prompts[6][1], 64); gen(prompts[2][1], 64)  # warm (1 code, 1 creative)
print(f"{'prompt':12} {'tok/s':>6} {'accLen':>7} {'per-position':>30}")
tot=[]
for name, p in prompts:
    m0 = metrics(); s = gen(p); m1 = metrics()
    ndr = m1["drafts"] - m0["drafts"]
    acc = m1["accepted"] - m0["accepted"]
    pos = sorted(set(m0["per_pos"]) | set(m1["per_pos"]))
    rates = [(m1["per_pos"].get(i,0)-m0["per_pos"].get(i,0))/ndr if ndr else 0 for i in pos]
    alen = acc/ndr + 1 if ndr else 0
    tot.append((s, alen))
    print(f"{name:12} {s:6.0f} {alen:7.3f}   {' '.join(f'{r:.2f}' for r in rates)}")
print(f"\nAVG tok/s {sum(x[0] for x in tot)/len(tot):.1f}  |  AVG accLen {sum(x[1] for x in tot)/len(tot):.3f}")
