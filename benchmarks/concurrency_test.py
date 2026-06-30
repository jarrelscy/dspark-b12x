#!/usr/bin/env python3
"""Decisive concurrency test: sustained 2-in-flight code load with frequent
short requests finishing (forcing batch condense), measure aggregate acceptance
and coherence. If the KV-slot fix works, acceptance stays ~single-stream (4.5);
if broken, it collapses. Also flags any incoherent/empty/errored output."""
import os, json, urllib.request, threading, time, re
KEY=os.environ.get("VLLM_API_KEY",""); MODEL=os.environ.get("BENCH_MODEL","deepseek/v4flash")
BASE="http://localhost:8001"
funcs="\n\n".join(f"def m_{i}(d):\n    return [x*{i%5+1} for x in d if x%{i%7+1}==0]" for i in range(400))
CODE=("```python\n"+funcs+"\n```\nWrite a complete production Python implementation of an "
      "LRU+TTL sharded async key-value store, full code with docstrings.")

def metrics():
    try:
        txt=urllib.request.urlopen(urllib.request.Request(BASE+"/metrics",headers={"Authorization":f"Bearer {KEY}"}),timeout=30).read().decode()
    except Exception: return (0.0,0.0)
    a=d=0.0
    for ln in txt.splitlines():
        if ln.startswith("#"): continue
        if "spec_decode_num_accepted_tokens_total" in ln: a=float(re.search(r"\s([\d.eE+]+)$",ln).group(1))
        elif "spec_decode_num_drafts_total" in ln: d=float(re.search(r"\s([\d.eE+]+)$",ln).group(1))
    return (a,d)

def chat(prompt,mx):
    body=json.dumps({"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":mx,"temperature":0.0}).encode()
    req=urllib.request.Request(BASE+"/v1/chat/completions",data=body,headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    try:
        d=json.loads(urllib.request.urlopen(req,timeout=600).read()); m=d["choices"][0]["message"]
        return {"ok":True,"text":(m.get("content") or m.get("reasoning_content") or ""),"toks":d["usage"]["completion_tokens"]}
    except Exception as e: return {"ok":False,"err":str(e)[:150]}

stop=False; bad=[]; n_long=0; n_short=0; lock=threading.Lock()
def long_worker():
    global n_long
    while not stop:
        r=chat(CODE,1200)
        with lock:
            n_long+=1
            if not r["ok"] or r["toks"]<50 or len(r["text"])<100: bad.append(("long",r))
def short_worker():  # frequent finishers -> force condense while long runs
    global n_short
    while not stop:
        r=chat("Say hi.",8)
        with lock:
            n_short+=1
            if not r["ok"]: bad.append(("short",r))
        time.sleep(0.1)

a0,d0=metrics()
threads=[threading.Thread(target=long_worker),threading.Thread(target=long_worker),threading.Thread(target=short_worker)]
for t in threads: t.start()
time.sleep(70); stop=True
for t in threads: t.join()
a1,d1=metrics()
da,dd=a1-a0,d1-d0
print(f"long reqs={n_long} short reqs={n_short} (2 long always in flight + short finishers)")
print(f"CONCURRENT accept_len = {da/dd+1:.3f} (drafts={dd:.0f})   [single-stream ref ~4.52]")
print(f"bad/incoherent/errored outputs: {len(bad)}")
for kind,r in bad[:5]: print("  ",kind, r.get("err") or f"toks={r.get('toks')} len={len(r.get('text',''))}")
print("VERDICT:", "PASS — acceptance healthy under concurrency, all coherent" if (dd>0 and da/dd+1>3.8 and not bad) else "CHECK — see above")
