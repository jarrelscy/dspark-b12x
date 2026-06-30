#!/usr/bin/env bash
# Sustained code load at matched ~37k ctx: fire N sequential code gens to keep the
# engine continuously decoding, so the 10s logger yields many clean decode windows.
set -u
KEY="${VLLM_API_KEY:?set VLLM_API_KEY}"
MODEL="${BENCH_MODEL:-deepseek/v4flash}"
NFUNC="${NFUNC:-400}"; MAXOUT="${MAXOUT:-4000}"; REQS="${REQS:-6}"
CTR="${BENCH_CONTAINER:-dspark-b12x}"

A0=$(curl -s "http://localhost:8001/metrics" -H "Authorization: Bearer $KEY" | awk '/spec_decode_num_accepted_tokens_total/&&!/#/{a=$2} /spec_decode_num_drafts_total/&&!/#/{d=$2} END{print a" "d}')
python3 - "$KEY" "$MODEL" "$NFUNC" "$MAXOUT" "$REQS" <<'PY' &
import json,urllib.request,sys
KEY,MODEL,NF,MX,REQS=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])
funcs="\n\n".join(
f"def process_module_{i}(data, config=None):\n    result=[]\n    for idx,item in enumerate(data):\n        if item%{i%7+1}==0: result.append(transform_{i}(item,idx))\n        else: result.append(item*{i%5+1})\n    return result\n\ndef transform_{i}(x,idx):\n    return (x<<{i%4})^(idx+{i})" for i in range(NF))
tasks=["a high-performance in-memory key-value store with TTL expiry, LRU eviction, sharding and async I/O",
       "a complete B-tree implementation with insert, delete, range scan and rebalancing",
       "an async connection pool with health checks, backoff, and circuit breaker",
       "a lock-free MPMC ring buffer with batching and backpressure",
       "a streaming JSON parser with incremental tokenization and schema validation",
       "a write-ahead log with segment rotation, fsync batching and crash recovery"]
for r in range(REQS):
    prompt=("Here is a large Python module:\n\n```python\n"+funcs+"\n```\n\n"
            f"Now write a complete, production-quality Python implementation of {tasks[r%len(tasks)]}, "
            "with full type hints and docstrings. Provide the entire code, class by class.")
    body={"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":MX,"temperature":0.0}
    req=urllib.request.Request("http://localhost:8001/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    d=json.loads(urllib.request.urlopen(req,timeout=900).read())
    print("req",r,"PROMPT",d["usage"]["prompt_tokens"],"OUTPUT",d["usage"]["completion_tokens"],flush=True)
PY
GP=$!
sleep 110
A1=$(curl -s "http://localhost:8001/metrics" -H "Authorization: Bearer $KEY" | awk '/spec_decode_num_accepted_tokens_total/&&!/#/{a=$2} /spec_decode_num_drafts_total/&&!/#/{d=$2} END{print a" "d}')
echo "=== decode windows (sustained code) ==="
docker logs --since 125s "$CTR" 2>&1 | grep -iE "generation throughput" | grep -E "Avg prompt throughput: 0.0" | grep -E "Running: 1" \
 | sed -E 's/.*generation throughput: ([0-9.]+) tokens.*/\1/' \
 | awk '{s+=$1;n++;if($1>mx)mx=$1;if(mn==""||$1<mn)mn=$1} END{if(n)printf "  windows=%d min=%.1f mean=%.1f max=%.1f tok/s\n",n,mn,s/n,mx; else print "  none"}'
awk -v a0="${A0%% *}" -v d0="${A0##* }" -v a1="${A1%% *}" -v d1="${A1##* }" 'BEGIN{da=a1-a0;dd=d1-d0; if(dd>0) printf "  accept_len=%.3f (drafts=%.0f)\n", da/dd+1, dd}'
wait $GP 2>/dev/null