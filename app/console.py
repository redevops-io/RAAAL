"""The portfolio operating console (served at /) — the live decision surface for quantify.club.

Reuses the agentic_os console contract (attention_priority ranking) via an in-process Aggregator, and
renders the three side-by-side objective decisions (min_risk | max_return_to_risk | max_total_return)
plus the current/no-action baseline, the discovery Attention Queue, a mission drawer with strategy-level
EXPLAIN, and the human paper-approval gate. Prominent DEMO / not-investment-advice framing throughout.
"""
from __future__ import annotations

import html
import json
from typing import Dict, List

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from src.config import DEMO_DISCLAIMER, OBJECTIVE_DESCRIPTIONS, OBJECTIVE_LABELS

from .api_investment import _engine

router = APIRouter(tags=["console"])

_TENANT = "ReDevOps Pilot Development"


def _fmt_pct(x) -> str:
    try:
        return f"{float(x):.1%}"
    except Exception:  # noqa: BLE001
        return "—"


def _top_holdings(weights: Dict[str, float], n: int = 5) -> List[tuple]:
    items = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    return [(t, w) for t, w in items if w > 0.005][:n]


def _objective_card(objective: str, plan: dict) -> str:
    label = OBJECTIVE_LABELS.get(objective, objective)
    desc = OBJECTIVE_DESCRIPTIONS.get(objective, "")
    sel = plan.get("selected_strategy_id", "—")
    alts = ", ".join(plan.get("alternative_strategy_ids", [])[:3]) or "—"
    et = plan.get("expected_tradeoffs", {})
    bm = plan.get("benchmark_metrics", {})
    binding = plan.get("binding_constraints", [])
    conf = plan.get("confidence", 0)
    bars = "".join(
        f'<div class="bar"><span class="tkr">{html.escape(t)}</span>'
        f'<span class="track"><span class="fill" style="width:{min(100, w*100):.0f}%"></span></span>'
        f'<span class="wv">{_fmt_pct(w)}</span></div>'
        for t, w in _top_holdings(plan.get("weights", {})))
    warn = ""
    if plan.get("abstained"):
        warn = '<div class="warn">Abstain suggested — does not beat holding cash on this objective.</div>'
    return f"""
    <div class="objcard">
      <div class="objhdr"><span class="objlabel">{html.escape(label)}</span>
        <span class="conf">conf {float(conf):.0%}</span></div>
      <div class="objdesc">{html.escape(desc)}</div>
      <div class="selrow"><span class="sellbl">Selected strategy</span>
        <span class="selstrat">{html.escape(str(sel))}</span></div>
      <div class="alt">alternatives: {html.escape(alts)}</div>
      <div class="metrics">
        <div><b>{_fmt_pct(et.get('exp_return'))}</b><span>exp return</span></div>
        <div><b>{_fmt_pct(et.get('exp_vol'))}</b><span>exp vol</span></div>
        <div><b>{float(et.get('sharpe',0)):.2f}</b><span>sharpe</span></div>
        <div><b>{_fmt_pct(bm.get('max_drawdown'))}</b><span>held-out DD</span></div>
      </div>
      <div class="bars">{bars}</div>
      <div class="binding">{'· '.join(html.escape(b) for b in binding) if binding else 'mandate satisfied'}</div>
      {warn}
    </div>"""


def _current_card(current: dict) -> str:
    m = current.get("metrics", {})
    bars = "".join(
        f'<div class="bar"><span class="tkr">{html.escape(t)}</span>'
        f'<span class="track"><span class="fill cash" style="width:{min(100, w*100):.0f}%"></span></span>'
        f'<span class="wv">{_fmt_pct(w)}</span></div>'
        for t, w in _top_holdings(current.get("weights", {})))
    return f"""
    <div class="objcard current">
      <div class="objhdr"><span class="objlabel">Current / no action</span><span class="conf">baseline</span></div>
      <div class="objdesc">The 4th comparison column — hold the current paper book. Always a valid choice.</div>
      <div class="selrow"><span class="sellbl">Position</span><span class="selstrat">as held</span></div>
      <div class="metrics">
        <div><b>{_fmt_pct(m.get('exp_return'))}</b><span>exp return</span></div>
        <div><b>{_fmt_pct(m.get('exp_vol'))}</b><span>exp vol</span></div>
        <div><b>{float(m.get('sharpe',0)):.2f}</b><span>sharpe</span></div>
        <div><b>—</b><span></span></div>
      </div>
      <div class="bars">{bars}</div>
    </div>"""


@router.get("/", response_class=HTMLResponse)
def console():
    eng = _engine()
    compare = eng.compare(refresh=True)
    disc = eng.discoveries()
    learn = eng.learning()
    plans = compare.get("plans", {})
    order = ["min_risk", "max_return_to_risk", "max_total_return"]
    cards = "".join(_objective_card(o, plans.get(o, {})) for o in order) + _current_card(compare.get("current", {}))

    queue_rows = "".join(
        f'<tr onclick="openMission()"><td>{i+1}</td><td>{html.escape(q["subject"])}</td>'
        f'<td><span class="pill">{html.escape(q["opportunity_class"])}</span></td>'
        f'<td class="num">{q["score"]:.3f}</td>'
        f'<td class="num">${q["expected_value"]:,.0f}</td>'
        f'<td class="num">{q["confidence"]:.0%}</td></tr>'
        for i, q in enumerate(disc.get("queue", [])[:12]))
    if not queue_rows:
        queue_rows = '<tr><td colspan="6" style="color:#667">Calm cycle — no material discoveries.</td></tr>'

    learn_pill = "shadow · learning" if learn.get("enabled") else "learning offline"
    promote = learn.get("promotion", "") if learn.get("enabled") else ""

    boot = json.dumps({"regime": compare.get("regime"), "snapshot_id": compare.get("snapshot_id")})
    page = _PAGE.format(
        tenant=html.escape(_TENANT), disclaimer=html.escape(DEMO_DISCLAIMER),
        regime=html.escape(str(compare.get("regime", "—"))), cards=cards, queue=queue_rows,
        learn_pill=html.escape(learn_pill), promote=html.escape(promote),
        ndisc=len(disc.get("queue", [])), boot=boot)
    # a live operating console must never be edge-cached (stale portfolio data)
    return HTMLResponse(page, headers={"Cache-Control": "no-store, max-age=0"})


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Investment Portfolio — operating console (DEMO)</title>
<style>
  :root{{--bg:#0b0f17;--panel:#121826;--line:#1f2937;--fg:#e5e7eb;--mut:#8b98ad;--acc:#5eead4;--warn:#f59e0b}}
  *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,sans-serif}}
  .wrap{{max-width:1180px;margin:0 auto;padding:22px 18px 60px}}
  h1{{margin:0;font-size:22px}} .sub{{color:var(--mut);margin:2px 0 0}}
  .demo{{background:#3a2d05;border-left:4px solid var(--warn);color:#fde68a;padding:10px 14px;border-radius:6px;margin:12px 0}}
  .row{{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:8px 0}}
  .chip{{background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:4px 12px;color:var(--mut);font-size:12px}}
  .chip b{{color:var(--fg)}}
  h2{{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:26px 0 10px}}
  .band{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
  @media(max-width:900px){{.band{{grid-template-columns:1fr 1fr}}}}
  .objcard{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}}
  .objcard.current{{opacity:.85}}
  .objhdr{{display:flex;justify-content:space-between;align-items:baseline}}
  .objlabel{{font-weight:700;color:var(--acc)}} .conf{{font-size:11px;color:var(--mut)}}
  .objdesc{{color:var(--mut);font-size:12px;margin:6px 0 10px;min-height:52px}}
  .selrow{{display:flex;justify-content:space-between;border-top:1px solid var(--line);padding-top:8px}}
  .sellbl{{color:var(--mut);font-size:12px}} .selstrat{{font-family:ui-monospace,monospace;color:#fff}}
  .alt{{color:var(--mut);font-size:11px;margin:2px 0 10px}}
  .metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:10px}}
  .metrics div{{background:#0e1420;border-radius:6px;padding:6px;text-align:center}}
  .metrics b{{display:block;font-size:14px}} .metrics span{{color:var(--mut);font-size:10px}}
  .bar{{display:flex;align-items:center;gap:6px;margin:3px 0}}
  .tkr{{width:64px;font-family:ui-monospace,monospace;font-size:11px;color:#cbd5e1}}
  .track{{flex:1;height:8px;background:#0e1420;border-radius:5px;overflow:hidden}}
  .fill{{display:block;height:100%;background:var(--acc)}} .fill.cash{{background:#64748b}}
  .wv{{width:44px;text-align:right;font-size:11px;color:var(--mut)}}
  .binding{{color:var(--mut);font-size:11px;margin-top:8px}}
  .warn{{color:#fca5a5;font-size:11px;margin-top:6px}}
  table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
  th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line);font-size:13px}}
  th{{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase}}
  tr[onclick]{{cursor:pointer}} tr[onclick]:hover{{background:#182234}}
  td.num{{text-align:right;font-variant-numeric:tabular-nums}}
  .pill{{background:#0e2a26;color:var(--acc);border-radius:999px;padding:2px 9px;font-size:11px}}
  .btn{{background:var(--acc);color:#04241f;border:0;border-radius:8px;padding:9px 16px;font-weight:700;cursor:pointer}}
  .btn.ghost{{background:transparent;color:var(--fg);border:1px solid var(--line)}}
  .actions{{margin:16px 0}}
  #drawer{{position:fixed;top:0;right:0;height:100%;width:min(560px,94vw);background:var(--panel);border-left:1px solid var(--line);
    transform:translateX(100%);transition:.25s;overflow:auto;padding:20px;box-shadow:-8px 0 30px #0008}}
  #drawer.open{{transform:none}} #drawer h3{{margin:0 0 6px}}
  .foot{{color:var(--mut);font-size:12px;border-top:1px solid var(--line);margin-top:28px;padding-top:10px}}
  .mono{{font-family:ui-monospace,monospace}}
  .exbranch{{border:1px solid var(--line);border-radius:8px;padding:10px;margin:8px 0}}
  .gate{{background:#0e2a26;border:1px solid #164e46;border-radius:8px;padding:12px;margin-top:14px}}
</style></head><body><div class="wrap">
  <h1>Investment Portfolio <span style="color:var(--mut);font-weight:400">— operating console</span></h1>
  <div class="sub">One console over RAAAL's research-backed engine. Tenant: <b>{tenant}</b></div>
  <div class="demo"><b>DEMO — not investment advice.</b> {disclaimer}</div>
  <div class="row">
    <span class="chip">regime <b>{regime}</b></span>
    <span class="chip">discoveries <b>{ndisc}</b></span>
    <span class="chip">learning <b>{learn_pill}</b></span>
    <span class="chip">{promote}</span>
    <span class="chip"><a href="/research" style="color:var(--acc);text-decoration:none">research dashboard →</a></span>
  </div>

  <h2>Objective decisions — three plans from one evidence snapshot (no averaging)</h2>
  <div class="band">{cards}</div>
  <div class="actions">
    <button class="btn" onclick="propose()">Open governed mission (propose paper rebalance)</button>
    <span style="color:var(--mut);font-size:12px;margin-left:8px">Every rebalance is human-approved and paper-only.</span>
  </div>

  <h2>Attention queue — discovered changes ranked</h2>
  <table><thead><tr><th>#</th><th>subject</th><th>class</th><th>score</th><th>exp. value</th><th>conf</th></tr></thead>
  <tbody>{queue}</tbody></table>

  <div class="foot">Aggregation: agentic_os.console (BFF) · RAAAL is the quantitative authority; the runtime
  selects among registered strategies, never fabricates allocations. Paper trading only — no external
  execution path exists.</div>
</div>
<div id="drawer"><button class="btn ghost" onclick="closeD()">Close</button><div id="dbody"></div></div>
<script>
const BOOT = {boot};
async function j(u,opt){{const r=await fetch(u,opt);return r.json();}}
function closeD(){{document.getElementById('drawer').classList.remove('open');}}
function openMission(){{propose();}}
async function propose(){{
  const m = await j('/api/investment/projects/default/missions/objective-compare',
    {{method:'POST',headers:{{'content-type':'application/json'}},body:JSON.stringify({{trigger:{{opportunity_class:'manual_review'}}}})}});
  const ex = await j('/api/investment/missions/'+m.mission_id+'/explain');
  renderDrawer(m, ex);
}}
function renderDrawer(m, ex){{
  const branches = (m.branches||[]).map(b=>`
    <div class="exbranch">
      <div><b>${{b.objective}}</b> → <span class="mono">${{b.selected_strategy_id}}</span></div>
      <div style="color:#8b98ad;font-size:12px">representation: ${{(b.representation||'—')}}
        · alternatives: ${{(b.alternative_strategy_ids||[]).join(', ')||'—'}}</div>
      <div style="margin-top:6px">
        <button class="btn" onclick="approve('${{m.mission_id}}','${{b.objective}}')">Approve (paper) — ${{b.objective}}</button>
      </div>
    </div>`).join('');
  document.getElementById('dbody').innerHTML = `
    <h3>Governed mission</h3>
    <div style="color:#8b98ad;font-size:12px">snapshot ${{m.snapshot_id}} · regime ${{m.regime}}</div>
    <div class="gate"><b>Human approval gate</b> — first and every paper rebalance requires explicit approval.
      Approving writes a <b>paper</b> order only; nothing is sent to a real venue.</div>
    ${{branches}}`;
  document.getElementById('drawer').classList.add('open');
}}
async function approve(mid, objective){{
  const r = await j('/api/investment/missions/'+mid+'/approve',
    {{method:'POST',headers:{{'content-type':'application/json'}},body:JSON.stringify({{objective}})}});
  alert(r.approved ? ('Paper rebalance recorded for '+objective+' (dispatched_externally='+r.dispatched_externally+'). No real order was placed.')
                   : ('Could not approve: '+(r.error||'unknown')));
}}
</script></body></html>"""
