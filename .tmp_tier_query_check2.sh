#!/usr/bin/env bash
TEST_VIDEO="big_bang_theory.mp4"
KUAVI="/home/berkin/Documents/academic_research/VideoRLM/.venv/bin/kuavi"
PY="/home/berkin/Documents/academic_research/VideoRLM/.venv/bin/python"
TIMEOUT_SECS=30

run() {
  local label="$1"
  local tier="$2"
  local query="$3"
  local result tier_used confidence top status

  result=$(timeout "$TIMEOUT_SECS" "$KUAVI" query --video "$TEST_VIDEO" --question "$query" --max-tier "$tier" --output-format jsonl 2>/dev/null || true)

  tier_used=$(printf '%s\n' "$result" | grep '"type": "cost"' | head -n1 | grep -o '"tier_used": [^,}]*' | grep -o '[0-9.]*' || true)
  confidence=$(printf '%s\n' "$result" | grep '"type": "result"' | head -n1 | grep -o '"confidence": [^,}]*' | grep -o '[0-9.]*' || true)

  top=$(printf '%s\n' "$result" | "$PY" -c 'import sys,json
lines=sys.stdin.read().splitlines()
res=None
for ln in lines:
    try:
        d=json.loads(ln)
    except Exception:
        continue
    if d.get("type")=="result":
        res=d
        break
if not res:
    print("-")
    raise SystemExit(0)
ts=res.get("timestamps") or []
if not ts:
    print("-")
    raise SystemExit(0)
f=ts[0]
if isinstance(f,dict):
    val=f.get("start",f.get("start_time",f.get("timestamp",f.get("time","-"))))
elif isinstance(f,(list,tuple)) and f:
    val=f[0]
else:
    val=f
try:
    sec=float(val)
    t=max(0,int(round(sec)))
    print(f"{t//60:02d}:{t%60:02d}")
except Exception:
    txt=str(val).strip(); p=txt.split(":")
    if len(p)==2 and all(x.isdigit() for x in p):
        print(f"{int(p[0]):02d}:{int(p[1]):02d}")
    elif len(p)==3 and all(x.isdigit() for x in p):
        h,m,s=map(int,p); tt=h*3600+m*60+s; print(f"{tt//60:02d}:{tt%60:02d}")
    else:
        print(txt)
' 2>/dev/null)

  [ -z "$tier_used" ] && tier_used="-"
  [ -z "$confidence" ] && confidence="-"
  [ -z "$top" ] && top="-"

  status="FAIL"
  [ "$top" != "-" ] && status="PASS"
  echo "$status [$label] tier=$tier_used conf=$confidence top=$top"
}

echo "=== TIER 1 — V-JEPA probes ==="
run "T1-temporal" 1 "what is happening at minute 3"
run "T1-action" 1 "what action is being performed at minute 5"
run "T1-coherence" 1 "is the motion at minute 7 natural and smooth"
run "T1-sport" 1 "what sport is being played"

echo "=== TIER 2 — LanguageBind search ==="
run "T2-visual" 2 "find the scene where someone picks up an object"
run "T2-audio" 2 "find the scene where two people are arguing"
run "T2-transcript" 2 "what is said at the beginning of the video"
run "T2-character" 2 "when does the main character first appear"
run "T2-sound" 2 "when does the music change"
run "T2-indoor" 2 "find where the indoor scene starts"

echo "=== TIER 2.5 — escalation check ==="
run "T25-ambiguous" 2 "find a tense moment"
run "T25-subtle" 2 "when does the background noise change"

echo "=== CONFIDENCE CHECK ==="
for item in "find the scene where someone picks up an object|2" "what is happening at minute 3|1"; do
  q="${item%|*}"; t="${item#*|}"
  out=$(timeout "$TIMEOUT_SECS" "$KUAVI" query --video "$TEST_VIDEO" --question "$q" --max-tier "$t" --output-format jsonl 2>/dev/null || true)
  conf=$(printf '%s\n' "$out" | grep '"type": "result"' | head -n1 | grep -o '"confidence": [^,}]*' | grep -o '[0-9.]*' || true)
  [ -z "$conf" ] && conf="0.000"
  "$PY" - <<PY
c=float("$conf")
status="PASS" if c>=0.2 else "WARN old low score"
print(f"{status} conf={c:.3f} query={'$q'[:40]}")
PY
done

echo "=== DONE ==="
