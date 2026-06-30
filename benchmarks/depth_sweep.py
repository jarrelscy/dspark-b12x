#!/usr/bin/env python3
"""Depth sweep with CODE context + CODE output (high acceptance) so the decode
column reflects realistic structured-generation speed vs context depth.
TTFT = first-token; decode = (completion-1)/(stream_end - first_token) [true wall]."""
import os, json, time, urllib.request
KEY=os.environ.get("VLLM_API_KEY",""); MODEL=os.environ.get("BENCH_MODEL","deepseek/v4flash")
BASE="http://localhost:8001"
# func counts chosen to land near 4k/32k/128k/512k/~950k ACTUAL prompt tokens
FUNCS=[int(x) for x in os.environ.get("FUNCS","55,440,1750,7000,13000").split(",")]
MAXOUT=int(os.environ.get("MAXOUT","800"))

def code_ctx(nf, tag):
    return (f"# module build {tag}\n"+"\n\n".join(
        f"def process_module_{i}(data, config=None):\n"
        f"    result = []\n"
        f"    for idx, item in enumerate(data):\n"
        f"        if item % {i%7+1} == 0:\n"
        f"            result.append(transform_{i}(item, idx))\n"
        f"        else:\n"
        f"            result.append(item * {i%5+1})\n"
        f"    return result\n\n"
        f"def transform_{i}(x, idx):\n    return (x << {i%4}) ^ (idx + {i})" for i in range(nf)))

def run(nf):
    prompt=("Here is a large Python module:\n\n```python\n"+code_ctx(nf,nf)+
            "\n```\n\nWrite a complete, production-quality Python implementation of an LRU+TTL "
            "sharded async key-value store with full type hints and docstrings; entire code, class by class.")
    body={"model":MODEL,"messages":[{"role":"user","content":prompt}],
          "max_tokens":MAXOUT,"temperature":0.0,"stream":True,"stream_options":{"include_usage":True}}
    req=urllib.request.Request(BASE+"/v1/chat/completions",data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    t0=time.perf_counter(); tfirst=None; usage=None
    resp=urllib.request.urlopen(req,timeout=1200)
    while True:
        line=resp.readline()
        if not line: break
        s=line.decode("utf-8","ignore").strip()
        if not s.startswith("data:"): continue
        d=s[5:].strip()
        if d=="[DONE]": break
        try: j=json.loads(d)
        except Exception: continue
        if j.get("usage"): usage=j["usage"]
        ch=j.get("choices") or []
        if ch and tfirst is None:
            dl=ch[0].get("delta",{})
            if dl.get("content") or dl.get("reasoning_content"): tfirst=time.perf_counter()
    tend=time.perf_counter()
    pt=usage["prompt_tokens"]; ct=usage["completion_tokens"]
    ttft=(tfirst-t0) if tfirst else (tend-t0); dec=(tend-tfirst) if tfirst else 0
    return pt,ct,ttft,(pt/ttft if ttft>0 else 0),((ct-1)/dec if dec>0 and ct>1 else 0)

print(f"model={MODEL}  (code context + code output, high acceptance)")
print(f"{'prompt_tok':>11} {'out_tok':>8} {'TTFT(s)':>8} {'prefill tok/s':>14} {'decode tok/s':>13}")
for nf in FUNCS:
    try:
        pt,ct,ttft,ptps,dtps=run(nf)
        print(f"{pt:>11} {ct:>8} {ttft:>8.2f} {ptps:>14.0f} {dtps:>13.1f}",flush=True)
    except Exception as e:
        print(f"  FUNCS={nf} ERROR: {str(e)[:120]}",flush=True)
