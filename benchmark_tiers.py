from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from kuavi.query_router import QueryRouter
from kuavi.tier_executors import execute_tier1, execute_tier2, execute_tier3
from kuavi.tiered_pipeline import TieredContext, _estimate_cost, load_or_build_index


BENCHMARK_QUESTIONS = [
    {"id": "q01", "question": "what action is happening at minute 3", "expected_tier": 1, "type": "temporal"},
    {"id": "q02", "question": "what sport is being played", "expected_tier": 1, "type": "classification"},
    {"id": "q03", "question": "what is the person doing at minute 5", "expected_tier": 1, "type": "temporal"},
    {"id": "q04", "question": "predict what happens next after minute 10", "expected_tier": 1, "type": "temporal"},
    {"id": "q05", "question": "is the motion at minute 7 natural and smooth", "expected_tier": 1, "type": "coherence"},
    {"id": "q06", "question": "find the scene where someone picks up an object", "expected_tier": 2, "type": "search"},
    {"id": "q07", "question": "when does the main character first appear", "expected_tier": 2, "type": "search"},
    {"id": "q08", "question": "find the scene where two people are talking", "expected_tier": 2, "type": "search"},
    {"id": "q09", "question": "when does the music change", "expected_tier": 2, "type": "search"},
    {"id": "q10", "question": "what is said at the beginning of the video", "expected_tier": 2, "type": "transcript"},
    {"id": "q11", "question": "find the scene where someone sits down", "expected_tier": 2, "type": "search"},
    {"id": "q12", "question": "at what timestamp does the scene change indoors", "expected_tier": 2, "type": "search"},
    {"id": "q13", "question": "summarize what happens in this video", "expected_tier": 3, "type": "summary"},
    {"id": "q14", "question": "explain the main storyline", "expected_tier": 3, "type": "explanation"},
    {"id": "q15", "question": "why does the character react that way", "expected_tier": 3, "type": "reasoning"},
    {"id": "q16", "question": "describe everything that happens in the first 5 minutes", "expected_tier": 3, "type": "description"},
    {"id": "q17", "question": "what caused the scene change at minute 8", "expected_tier": 3, "type": "reasoning"},
    {"id": "q18", "question": "tell me about the relationships between characters", "expected_tier": 3, "type": "explanation"},
    {"id": "q19", "question": "what is the overall mood of this video", "expected_tier": 3, "type": "description"},
    {"id": "q20", "question": "how does the narrative develop throughout", "expected_tier": 3, "type": "explanation"},
]


@dataclass
class TierRunResult:
    elapsed_ms: int
    answer_preview: str
    answer_full: str
    confidence: float
    escalated: bool
    llm_calls: int
    estimated_usd: float
    error: str | None
    answered: bool


def _json_escape(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _pct(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return (numer / denom) * 100.0


def _as_bool_dot(value: bool) -> str:
    return '<span class="dot dot-yes"></span>' if value else '<span class="dot dot-no"></span>'


def _resolve_video_path(video_path: str) -> Path:
    p = Path(video_path)
    if p.exists():
        return p
    media_candidate = Path("media") / video_path
    if media_candidate.exists():
        return media_candidate
    return p


async def _load_context(video_path: str) -> TieredContext:
    resolved = _resolve_video_path(video_path)
    if resolved.exists():
        return await load_or_build_index(str(resolved), index_mode="fast", asr_model="faster-whisper/base")

    sidecar = resolved.with_suffix(".kuavi")
    if sidecar.exists():
        from kuavi.indexer import VideoIndex

        index = VideoIndex.load(sidecar)
        return TieredContext(index=index, video_path=str(resolved))

    raise FileNotFoundError(
        f"Video not found: {video_path} (also checked {resolved} and sidecar {sidecar})"
    )


def _forced_routing(router: QueryRouter, question: str, tier: int) -> dict[str, Any]:
    base = router.classify(question)
    output_format = base.get("output_format", "text")

    if tier == 1:
        tools = router._suggest_tier1_tools(question.lower())
        return {
            "tier": 1,
            "reason": "forced-tier-1",
            "suggested_tools": tools or ["orient"],
            "output_format": output_format,
        }
    if tier == 2:
        tools = router._suggest_tier2_tools(question.lower())
        return {
            "tier": 2,
            "reason": "forced-tier-2",
            "suggested_tools": tools or ["search_all"],
            "output_format": output_format,
        }
    return {
        "tier": 3,
        "reason": "forced-tier-3",
        "suggested_tools": ["full_agent"],
        "output_format": "text",
    }


def _answered_proxy(tier: int, result: dict[str, Any], error: str | None) -> bool:
    if error:
        return False
    answer = str(result.get("answer", "") or "")
    if tier == 1:
        return float(result.get("confidence", 0.0)) >= 0.5
    if tier == 2:
        ts = result.get("timestamps") or []
        return bool(ts) or len(answer) > 50
    return len(answer) > 100


async def _run_single_tier(
    *,
    ctx: TieredContext,
    question: str,
    router: QueryRouter,
    tier: int,
    skip_tier3: bool,
    model: str,
    backend: str,
) -> TierRunResult:
    routing = _forced_routing(router, question, tier)
    start = time.monotonic()

    if tier == 3 and skip_tier3:
        return TierRunResult(
            elapsed_ms=0,
            answer_preview="skipped",
            answer_full="skipped",
            confidence=0.0,
            escalated=False,
            llm_calls=0,
            estimated_usd=0.0,
            error=None,
            answered=False,
        )

    try:
        if tier == 1:
            result = await execute_tier1(ctx, question, routing)
        elif tier == 2:
            result = await execute_tier2(ctx.index, question, routing)
        else:
            result = await execute_tier3(
                ctx,
                question,
                routing,
                model=model,
                backend=backend,
                tier2_result=None,
            )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        answer = str(result.get("answer", "") or "")
        err = None
        answered = _answered_proxy(tier, result, err)
        estimated_usd = float(_estimate_cost(result)) if tier == 3 else 0.0
        return TierRunResult(
            elapsed_ms=elapsed_ms,
            answer_preview=answer[:200],
            answer_full=answer,
            confidence=float(result.get("confidence", 0.0) or 0.0),
            escalated=bool(result.get("escalate", False)),
            llm_calls=int(result.get("llm_calls", 0) or 0),
            estimated_usd=estimated_usd,
            error=err,
            answered=answered,
        )
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return TierRunResult(
            elapsed_ms=elapsed_ms,
            answer_preview="",
            answer_full="",
            confidence=0.0,
            escalated=False,
            llm_calls=0,
            estimated_usd=0.0,
            error=str(exc),
            answered=False,
        )


def generate_html_report(results: list[dict[str, Any]], output_path: str, video_path: str) -> None:
    n = len(results)
    tier1_times = [float(r["tier1"]["elapsed_ms"]) for r in results]
    tier2_times = [float(r["tier2"]["elapsed_ms"]) for r in results]
    tier3_times = [float(r["tier3"]["elapsed_ms"]) for r in results if not r["tier3"].get("skipped", False)]

    t1_avg = _mean(tier1_times)
    t2_avg = _mean(tier2_times)
    t3_avg = _mean(tier3_times)

    t1_answered = sum(1 for r in results if r["tier1"]["answered"])
    t2_answered = sum(1 for r in results if r["tier2"]["answered"])
    t3_answered = sum(1 for r in results if r["tier3"]["answered"])

    t1_answer_rate = _pct(t1_answered, n)
    t2_answer_rate = _pct(t2_answered, n)
    t3_answer_rate = _pct(t3_answered, n)

    router_correct = sum(1 for r in results if r["router_correct"])
    router_incorrect = n - router_correct
    router_acc_pct = _pct(router_correct, n)

    llm_calls_by_q = [int(r["tier3"]["llm_calls"]) for r in results]
    costs_by_q = [float(r["tier3"]["estimated_usd"]) for r in results]
    tier3_total_cost = sum(costs_by_q)

    e12 = sum(1 for r in results if r["tier1"]["escalated"])
    e23 = sum(1 for r in results if r["tier2"]["escalated"])
    no_escalation = max(0, n - e12 - e23)

    if n > 0:
        measured_tier3_avg_cost = tier3_total_cost / n
    else:
        measured_tier3_avg_cost = 0.0
    full_llm_baseline = measured_tier3_avg_cost * n
    tiered_cost = tier3_total_cost
    savings_pct = 0.0
    if full_llm_baseline > 0:
        savings_pct = ((full_llm_baseline - tiered_cost) / full_llm_baseline) * 100.0

    rows = []
    accordions = []
    for r in results:
        rid = r["id"]
        router_cell_class = "router-ok" if r["router_correct"] else "router-bad"
        t3_cost_cell_class = "cost-hot" if float(r["tier3"]["estimated_usd"]) > 0 else ""
        rows.append(
            "<tr>"
            f"<td>{rid}</td>"
            f"<td>{r['question']}</td>"
            f"<td>{r['type']}</td>"
            f"<td>T{r['expected_tier']}</td>"
            f"<td class='{router_cell_class}'>T{r['router_tier']}</td>"
            f"<td>{'✓' if r['router_correct'] else '✗'}</td>"
            f"<td>{r['tier1']['elapsed_ms']}ms</td>"
            f"<td>{_as_bool_dot(bool(r['tier1']['answered']))}</td>"
            f"<td>{r['tier2']['elapsed_ms']}ms</td>"
            f"<td>{_as_bool_dot(bool(r['tier2']['answered']))}</td>"
            f"<td>{r['tier3']['elapsed_ms']}ms</td>"
            f"<td>{r['tier3']['llm_calls']}</td>"
            f"<td class='{t3_cost_cell_class}'>${float(r['tier3']['estimated_usd']):.4f}</td>"
            "</tr>"
        )

        accordions.append(
            "<details class='acc-item'>"
            f"<summary>{rid} — {r['question']}</summary>"
            "<div class='acc-body'>"
            f"<p><strong>Question:</strong> {r['question']}</p>"
            f"<p><strong>Tier 1:</strong> {r['tier1']['answer_full'] or r['tier1']['error'] or ('escalated' if r['tier1']['escalated'] else 'N/A')}</p>"
            f"<p><strong>Tier 2:</strong> {r['tier2']['answer_full'] or r['tier2']['error'] or ('escalated' if r['tier2']['escalated'] else 'N/A')}</p>"
            f"<p><strong>Tier 3:</strong> {r['tier3']['answer_full'] or r['tier3']['error'] or ('skipped' if r['tier3'].get('skipped') else 'N/A')}</p>"
            "</div>"
            "</details>"
        )

    chart_labels = [r["id"] for r in results]

    html = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>VideoRLM Tier Benchmark Report</title>
  <style>
    :root {{ --bg:#0b0f17; --panel:#121826; --text:#e6edf7; --muted:#9fb0cc; --ok:#2ecc71; --bad:#e74c3c; --amber:#f39c12; --blue:#3498db; --green:#2ecc71; --orange:#e67e22; }}
    body {{ margin:0; font-family:Inter,Arial,sans-serif; background:var(--bg); color:var(--text); }}
    .wrap {{ max-width:1300px; margin:0 auto; padding:24px; }}
    .panel {{ background:var(--panel); border:1px solid #1f2a40; border-radius:14px; padding:16px; margin-bottom:16px; }}
    h1,h2,h3 {{ margin:0 0 12px; }}
    .badges {{ display:flex; gap:10px; flex-wrap:wrap; }}
    .badge {{ padding:8px 12px; border-radius:999px; background:#1a2436; border:1px solid #2a3a57; color:var(--text); font-size:13px; }}
    .cards {{ display:grid; grid-template-columns:repeat(3,minmax(250px,1fr)); gap:12px; }}
    .card {{ background:#111a2a; border:1px solid #223252; border-radius:10px; padding:14px; }}
    .chart-grid {{ display:grid; grid-template-columns:repeat(2,minmax(300px,1fr)); gap:14px; }}
    .chart-cell {{ background:#111a2a; border:1px solid #223252; border-radius:10px; padding:12px; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; }}
    th,td {{ border:1px solid #23324f; padding:6px 7px; vertical-align:top; }}
    th {{ background:#18253c; position:sticky; top:0; z-index:2; }}
    .router-ok {{ background:rgba(46,204,113,.20); }}
    .router-bad {{ background:rgba(231,76,60,.20); }}
    .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; }}
    .dot-yes {{ background:var(--ok); }}
    .dot-no {{ background:var(--bad); }}
    .cost-hot {{ background:rgba(243,156,18,.22); }}
    .acc-item {{ border:1px solid #223252; border-radius:8px; margin-bottom:8px; background:#111a2a; }}
    .acc-item summary {{ cursor:pointer; padding:10px 12px; font-weight:600; }}
    .acc-body {{ padding:0 12px 10px; color:#d2def2; }}
    .muted {{ color:var(--muted); }}
  </style>
</head>
<body>
  <div class='wrap'>
    <section class='panel'>
      <h1>VideoRLM Tier Benchmark Report</h1>
      <p class='muted'>Video: {video_path} • Run at: {datetime.now().isoformat(timespec='seconds')} • Total questions: {n}</p>
      <div class='badges'>
        <span class='badge'>Tier 1 avg: {t1_avg:.1f}ms</span>
        <span class='badge'>Tier 2 avg: {t2_avg:.1f}ms</span>
        <span class='badge'>Tier 3 avg: {t3_avg:.1f}ms</span>
      </div>
    </section>

    <section class='panel'>
      <h2>Tier Explanation</h2>
      <div class='cards'>
        <div class='card'>
          <h3>TIER 1 — World Model (V-JEPA)</h3>
          <p><strong>Cost:</strong> $0.00 per query</p>
          <p>How it works: Uses V-JEPA 2, a video world model trained by Meta to predict masked spacetime patches. It understands temporal structure, action sequences, and physical coherence directly from video features — no language model involved.</p>
          <p><strong>Best for:</strong> Temporal questions, action classification, coherence checks.</p>
          <p><strong>Limitation:</strong> Cannot produce free-form text answers.</p>
        </div>
        <div class='card'>
          <h3>TIER 2 — Semantic Embeddings</h3>
          <p><strong>Cost:</strong> $0.00 per query</p>
          <p>How it works: Uses SigLIP 2 (visual) and Gemma (text) embedding models to perform semantic similarity search over indexed video segments and transcript. Finds the most relevant scenes by meaning, not keyword matching.</p>
          <p><strong>Best for:</strong> Scene search, transcript lookup, multiple choice questions.</p>
          <p><strong>Limitation:</strong> Cannot reason across multiple scenes or synthesize answers.</p>
        </div>
        <div class='card'>
          <h3>TIER 3 — LLM Agent (Last Resort)</h3>
          <p><strong>Cost:</strong> ~$0.01–0.05 per query</p>
          <p>How it works: Routes to a full LLM (GPT-4o, Claude, Gemini) with access to all 31 kuavi tools. The agent autonomously decides which tools to call, synthesizes results across multiple tool calls, and produces a free-form natural language answer.</p>
          <p><strong>Best for:</strong> Summarization, multi-hop reasoning, open-ended questions.</p>
          <p><strong>Limitation:</strong> Expensive, slower, non-deterministic.</p>
        </div>
      </div>
    </section>

    <section class='panel'>
      <h2>Summary Charts</h2>
      <div class='chart-grid'>
        <div class='chart-cell'><h3>Average Response Time by Tier</h3><canvas id='chartTime'></canvas></div>
        <div class='chart-cell'><h3>Answer Rate by Tier</h3><canvas id='chartAnswer'></canvas></div>
        <div class='chart-cell'><h3>Router Classification Accuracy</h3><canvas id='chartRouter'></canvas></div>
        <div class='chart-cell'><h3>Total LLM Calls & Cost</h3><canvas id='chartCalls'></canvas></div>
        <div class='chart-cell'><h3>Escalation Rate</h3><canvas id='chartEscalation'></canvas></div>
      </div>
    </section>

    <section class='panel'>
      <h2>Per-Question Results</h2>
      <div style='overflow:auto; max-height:520px'>
        <table>
          <thead>
            <tr>
              <th>ID</th><th>Question</th><th>Type</th><th>Expected Tier</th><th>Router Said</th><th>Router ✓/✗</th>
              <th>T1 Time</th><th>T1 Answered</th><th>T2 Time</th><th>T2 Answered</th>
              <th>T3 Time</th><th>T3 LLM Calls</th><th>T3 Cost USD</th>
            </tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </section>

    <section class='panel'>
      <h2>Raw Answers Accordion</h2>
      {''.join(accordions)}
    </section>

    <section class='panel'>
      <h2>Conclusions</h2>
      <p>Tier 1 answered {t1_answer_rate:.1f}% of queries at $0 cost.</p>
      <p>Tier 2 answered {t2_answer_rate:.1f}% of queries at $0 cost.</p>
      <p>Tier 3 answered {t3_answer_rate:.1f}% of queries at avg ${measured_tier3_avg_cost:.4f} per query.</p>
      <p>The router correctly classified {router_correct}/{n} queries ({router_acc_pct:.1f}%).</p>
      <p>Estimated cost savings vs full LLM baseline: running all {n} queries through Tier 3 would cost ~${full_llm_baseline:.4f}. Tiered routing reduced this to ~${tiered_cost:.4f} ({savings_pct:.1f}% savings).</p>
    </section>
  </div>

  <script src='https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js'></script>
  <script>
    const green = '#2ecc71';
    const blue = '#3498db';
    const orange = '#e67e22';
    const amber = '#f39c12';

    new Chart(document.getElementById('chartTime'), {{
      type: 'bar',
      data: {{
        labels: ['Tier 1','Tier 2','Tier 3'],
        datasets: [{{ data: [{t1_avg:.3f},{t2_avg:.3f},{t3_avg:.3f}], backgroundColor:[green,blue,orange] }}]
      }},
      options: {{ plugins: {{ legend: {{ display:false }} }} }}
    }});

    new Chart(document.getElementById('chartAnswer'), {{
      type: 'bar',
      data: {{
        labels: ['Tier 1','Tier 2','Tier 3'],
        datasets: [{{ data: [{t1_answer_rate:.3f},{t2_answer_rate:.3f},{t3_answer_rate:.3f}], backgroundColor:[green,blue,orange] }}]
      }},
      options: {{ scales: {{ y: {{ min:0, max:100 }} }}, plugins: {{ legend: {{ display:false }} }} }}
    }});

    new Chart(document.getElementById('chartRouter'), {{
      type: 'doughnut',
      data: {{ labels:['Correct','Incorrect'], datasets:[{{ data:[{router_correct},{router_incorrect}], backgroundColor:[green,'#e74c3c'] }}] }}
    }});

    new Chart(document.getElementById('chartCalls'), {{
      type: 'bar',
      data: {{
        labels: {_json_escape(chart_labels)},
        datasets: [{{ label:'LLM Calls', data:{_json_escape(llm_calls_by_q)}, backgroundColor: amber }}]
      }}
    }});

    new Chart(document.getElementById('chartEscalation'), {{
      type: 'doughnut',
      data: {{ labels:['T1→T2','T2→T3','No escalation'], datasets:[{{ data:[{e12},{e23},{no_escalation}], backgroundColor:[blue,orange,green] }}] }}
    }});
  </script>
</body>
</html>
"""

    Path(output_path).write_text(html, encoding="utf-8")


def run_benchmark(
    video_path: str = "big_bang_theory.mp4",
    output_html: str = "benchmark_report.html",
    skip_tier3: bool = False,
    model: str = "gpt-4o",
    backend: str = "openai",
) -> None:
    print(f"VideoRLM Tier Benchmark — {video_path}")
    print("Preparing index/context...")

    router = QueryRouter()
    ctx = asyncio.run(_load_context(video_path))
    out_jsonl = Path("benchmark_results.jsonl")
    out_jsonl.write_text("", encoding="utf-8")

    results: list[dict[str, Any]] = []

    for item in BENCHMARK_QUESTIONS:
        qid = item["id"]
        question = item["question"]
        expected_tier = int(item["expected_tier"])

        routed = router.classify(question)
        router_tier = int(routed.get("tier", 2))
        router_correct = router_tier == expected_tier

        t1 = asyncio.run(
            _run_single_tier(
                ctx=ctx,
                question=question,
                router=router,
                tier=1,
                skip_tier3=skip_tier3,
                model=model,
                backend=backend,
            )
        )
        t2 = asyncio.run(
            _run_single_tier(
                ctx=ctx,
                question=question,
                router=router,
                tier=2,
                skip_tier3=skip_tier3,
                model=model,
                backend=backend,
            )
        )
        t3 = asyncio.run(
            _run_single_tier(
                ctx=ctx,
                question=question,
                router=router,
                tier=3,
                skip_tier3=skip_tier3,
                model=model,
                backend=backend,
            )
        )

        record = {
            "id": qid,
            "question": question,
            "type": item["type"],
            "expected_tier": expected_tier,
            "router_tier": router_tier,
            "router_correct": router_correct,
            "tier1": {
                "elapsed_ms": t1.elapsed_ms,
                "answer_preview": t1.answer_preview,
                "answer_full": t1.answer_full,
                "confidence": t1.confidence,
                "escalated": t1.escalated,
                "llm_calls": 0,
                "estimated_usd": 0.0,
                "error": t1.error,
                "answered": t1.answered,
            },
            "tier2": {
                "elapsed_ms": t2.elapsed_ms,
                "answer_preview": t2.answer_preview,
                "answer_full": t2.answer_full,
                "confidence": t2.confidence,
                "escalated": t2.escalated,
                "llm_calls": 0,
                "estimated_usd": 0.0,
                "error": t2.error,
                "answered": t2.answered,
            },
            "tier3": {
                "elapsed_ms": t3.elapsed_ms,
                "answer_preview": t3.answer_preview,
                "answer_full": t3.answer_full,
                "confidence": t3.confidence,
                "escalated": t3.escalated,
                "llm_calls": t3.llm_calls,
                "estimated_usd": t3.estimated_usd,
                "error": t3.error,
                "answered": t3.answered,
                "skipped": bool(skip_tier3),
            },
        }

        with out_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        results.append(record)

        t3_label = "skipped" if skip_tier3 else f"{t3.elapsed_ms}ms"
        print(
            f"[{qid}] Routing... T{router_tier} {'✓' if router_correct else '✗'} | "
            f"T1: {t1.elapsed_ms}ms | T2: {t2.elapsed_ms}ms | T3: {t3_label}"
        )

    generate_html_report(results, output_html, video_path)
    print(f"Report saved to {output_html}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="big_bang_theory.mp4")
    parser.add_argument("--output", default="benchmark_report.html")
    parser.add_argument("--skip-tier3", action="store_true", help="Skip Tier 3 LLM calls (saves API cost)")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--backend", default="openai")
    args = parser.parse_args()
    run_benchmark(args.video, args.output, args.skip_tier3, args.model, args.backend)