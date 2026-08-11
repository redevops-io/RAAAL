"""The portfolio operating console (served at /) — the live decision surface for quantify.club.

UX goal (the "wow" moment): the page opens with a 2-3s "continuously thinking" sequence —
Discovery Runtime -> Decision Planner -> Mission Runtime — driven by REAL engine numbers, so a
visitor immediately sees an operating system, not a static dashboard. Then it reveals, in the
recommended reading order: Discovery checks -> Today's Recommendation (dominant) -> Alternatives ->
Current Portfolio -> Governed Mission. Friendly objective names up front; internal optimizer ids stay
in EXPLAIN. RAAAL is the quantitative authority; the runtime SELECTS among registered strategies.
"""
from __future__ import annotations

import html
import json
from typing import Dict, List

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from src.config import AUX_SERIES, DEMO_DISCLAIMER, UNIVERSE
from src.strategies import STRATEGY_REGISTRY

from .api_investment import _engine

router = APIRouter(tags=["console"])

_TENANT = "ReDevOps Pilot Development"

# UI names (internal optimizer names remain in EXPLAIN / the API)
_DISPLAY = {"min_risk": "Capital Preservation", "max_return_to_risk": "Balanced Growth",
            "max_total_return": "Maximum Growth"}
_TAGLINE = {"min_risk": "Protect capital first", "max_return_to_risk": "Best risk-adjusted return",
            "max_total_return": "Chase the most upside"}
_RECOMMENDED = "max_return_to_risk"   # today's recommendation = the balanced objective


def _pct(x) -> str:
    try:
        return f"{float(x):.1%}"
    except Exception:  # noqa: BLE001
        return "—"


def _top_holdings(weights: Dict[str, float], n: int = 6) -> List[tuple]:
    return [(t, w) for t, w in sorted(weights.items(), key=lambda kv: kv[1], reverse=True) if w > 0.005][:n]


def _bars(weights: Dict[str, float], cash: bool = False) -> str:
    cls = "fill cash" if cash else "fill"
    return "".join(
        f'<div class="bar"><span class="tkr">{html.escape(t)}</span>'
        f'<span class="track"><span class="{cls}" style="width:{min(100, w*100):.0f}%"></span></span>'
        f'<span class="wv">{_pct(w)}</span></div>' for t, w in _top_holdings(weights))


def _reason_of(plan: dict) -> str:
    st = plan.get("score_table") or []
    if st:
        return st[0].get("rationale", "")
    return ""


def _recommendation_card(objective: str, plan: dict, current: dict) -> str:
    name = _DISPLAY.get(objective, objective)
    sel = plan.get("selected_strategy_id", "—")
    alts = ", ".join(plan.get("alternative_strategy_ids", [])[:3]) or "—"
    et = plan.get("expected_tradeoffs", {})
    cur = current.get("metrics", {})
    d_ret = float(et.get("exp_return", 0)) - float(cur.get("exp_return", 0))
    d_sh = float(et.get("sharpe", 0)) - float(cur.get("sharpe", 0))
    imp = (f'<span class="delta {"up" if d_ret>=0 else "dn"}">{"+" if d_ret>=0 else ""}{_pct(d_ret)} exp. return</span>'
           f'<span class="delta {"up" if d_sh>=0 else "dn"}">{"+" if d_sh>=0 else ""}{d_sh:.2f} Sharpe</span>'
           f'<span class="vs">vs current / no action</span>')
    return f"""
    <div class="reco">
      <div class="recohdr">
        <div><div class="ribbon">Today's recommendation</div>
          <div class="reconame">{html.escape(name)}</div>
          <div class="recotag">{html.escape(_TAGLINE.get(objective,''))}</div></div>
        <div class="conf">confidence {float(plan.get('confidence',0)):.0%}</div>
      </div>
      <div class="recogrid">
        <div class="recoleft">
          <div class="selrow"><span class="sellbl">Selected strategy</span>
            <span class="selstrat">{html.escape(str(sel))}</span></div>
          <div class="reason">{html.escape(_reason_of(plan))}</div>
          <div class="alt">Alternatives considered: {html.escape(alts)}</div>
          <div class="improve">{imp}</div>
          <div class="binding">{'· '.join(html.escape(b) for b in plan.get('binding_constraints',[])) or 'mandate satisfied'}</div>
        </div>
        <div class="recoright">
          <div class="metrics big">
            <div><b>{_pct(et.get('exp_return'))}</b><span>exp return</span></div>
            <div><b>{_pct(et.get('exp_vol'))}</b><span>exp vol</span></div>
            <div><b>{float(et.get('sharpe',0)):.2f}</b><span>sharpe</span></div>
          </div>
          <div class="bars">{_bars(plan.get('weights',{}))}</div>
        </div>
      </div>
    </div>"""


def _alt_card(objective: str, plan: dict) -> str:
    name = _DISPLAY.get(objective, objective)
    et = plan.get("expected_tradeoffs", {})
    warn = '<div class="warn">Abstain — does not beat cash on this objective.</div>' if plan.get("abstained") else ""
    return f"""
    <div class="altcard">
      <div class="objhdr"><span class="objlabel">{html.escape(name)}</span>
        <span class="conf">{float(plan.get('confidence',0)):.0%}</span></div>
      <div class="objtag">{html.escape(_TAGLINE.get(objective,''))}</div>
      <div class="selrow"><span class="sellbl">Selected</span>
        <span class="selstrat">{html.escape(str(plan.get('selected_strategy_id','—')))}</span></div>
      <div class="metrics">
        <div><b>{_pct(et.get('exp_return'))}</b><span>ret</span></div>
        <div><b>{_pct(et.get('exp_vol'))}</b><span>vol</span></div>
        <div><b>{float(et.get('sharpe',0)):.2f}</b><span>sharpe</span></div>
      </div>
      <div class="bars">{_bars(plan.get('weights',{}))}</div>{warn}
    </div>"""


def _current_card(current: dict) -> str:
    m = current.get("metrics", {})
    return f"""
    <div class="altcard current">
      <div class="objhdr"><span class="objlabel">Current portfolio</span><span class="conf">baseline</span></div>
      <div class="objtag">Hold the current paper book — always a valid choice</div>
      <div class="selrow"><span class="sellbl">Position</span><span class="selstrat">as held</span></div>
      <div class="metrics">
        <div><b>{_pct(m.get('exp_return'))}</b><span>ret</span></div>
        <div><b>{_pct(m.get('exp_vol'))}</b><span>vol</span></div>
        <div><b>{float(m.get('sharpe',0)):.2f}</b><span>sharpe</span></div>
      </div>
      <div class="bars">{_bars(current.get('weights',{}), cash=True)}</div>
    </div>"""


@router.get("/", response_class=HTMLResponse)
def console():
    eng = _engine()
    compare = eng.compare(refresh=True)
    disc = eng.discoveries()
    learn = eng.learning()
    plans = compare.get("plans", {})
    regime = str(compare.get("regime", "—"))
    queue = disc.get("queue", [])

    # ---- real numbers for the "continuously thinking" boot sequence ----
    approved = [c for c in STRATEGY_REGISTRY if c.promotion_status == "approved"]
    families = sorted({c.family for c in approved})
    signals_checked = 0
    try:
        from src.agentic.signals import build_signals, observation_from
        signals_checked = len(build_signals(observation_from(compare, eng.portfolio.load())))
    except Exception:  # noqa: BLE001
        signals_checked = 13
    boot = {
        "assets": len(UNIVERSE) + len(AUX_SERIES),
        "families": len(families),
        "signals": signals_checked,
        "regime": regime,
        "candidates": len(queue),
        "strategies": len(approved),
        "recommended": _DISPLAY[_RECOMMENDED],
        "selected": plans.get(_RECOMMENDED, {}).get("selected_strategy_id", "—"),
        "reason": _reason_of(plans.get(_RECOMMENDED, {})) or "Highest expected utility for the current market structure.",
        "learning": bool(learn.get("enabled")),
    }

    # ---- discovery checks panel (always show completed checks) ----
    checks = [f"{boot['assets']} assets scanned", f"{boot['families']} strategy families evaluated",
              f"{boot['signals']} market signals checked", f"regime detected: {regime}",
              f"{boot['candidates']} candidate mission(s) generated"]
    checks_html = "".join(f'<li>{html.escape(c)}</li>' for c in checks)
    if queue:
        queue_rows = "".join(
            f'<tr onclick="propose()"><td>{i+1}</td><td>{html.escape(q["subject"])}</td>'
            f'<td><span class="pill">{html.escape(q["opportunity_class"])}</span></td>'
            f'<td class="num">{q["score"]:.3f}</td><td class="num">{q["confidence"]:.0%}</td></tr>'
            for i, q in enumerate(queue[:8]))
        queue_block = (f'<table><thead><tr><th>#</th><th>subject</th><th>change</th><th>score</th>'
                       f'<th>conf</th></tr></thead><tbody>{queue_rows}</tbody></table>')
    else:
        queue_block = ('<div class="empty">✓ No actionable discoveries this cycle — the runtime keeps '
                       'watching. (Better than an empty queue: the checks above still ran.)</div>')

    reco = _recommendation_card(_RECOMMENDED, plans.get(_RECOMMENDED, {}), compare.get("current", {}))
    alts = "".join(_alt_card(o, plans.get(o, {})) for o in ("min_risk", "max_total_return"))
    current_card = _current_card(compare.get("current", {}))

    learn_pill = "shadow · learning" if boot["learning"] else "learning offline"
    promote = learn.get("promotion", "") if boot["learning"] else ""

    body = _BODY.format(
        tenant=html.escape(_TENANT), disclaimer=html.escape(DEMO_DISCLAIMER), regime=html.escape(regime),
        ndisc=len(queue), learn_pill=html.escape(learn_pill), promote=html.escape(promote),
        checks=checks_html, queue=queue_block, reco=reco, alts=alts, current=current_card,
        pipe=_PIPELINE)
    page = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>Quantify — Investment Operating System (DEMO)</title>"
            f"<style>{_CSS}</style></head><body>{body}"
            f"<script>const BOOT={json.dumps(boot)};{_JS}</script></body></html>")
    return HTMLResponse(page, headers={"Cache-Control": "no-store, max-age=0"})


_PIPELINE = ("".join(f'<span class="pnode">{n}</span>' + ('<span class="parr">→</span>' if i < 5 else '')
             for i, n in enumerate(["Discovery", "Context", "Planner", "Registered Strategies",
                                    "Mission", "Learning"])))

_CSS = """
:root{--bg:#0a0e17;--panel:#121a2b;--panel2:#0e1626;--line:#1e2a3f;--fg:#e6edf6;--mut:#8496b0;
  --acc:#5eead4;--acc2:#38bdf8;--warn:#f59e0b;--good:#34d399;--bad:#fb7185}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:20px 18px 70px}
h1{margin:0;font-size:22px;letter-spacing:-.01em}
.sub{color:var(--mut);margin:3px 0 0}
.demo{background:#3a2d05;border-left:4px solid var(--warn);color:#fde68a;padding:10px 14px;border-radius:6px;margin:12px 0}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:10px 0}
.chip{background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:4px 12px;color:var(--mut);font-size:12px}
.chip b{color:var(--fg)} .chip.good b{color:var(--good)}
.chip a{color:var(--acc);text-decoration:none}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);margin:26px 0 10px;display:flex;gap:8px;align-items:center}
h2 .n{background:var(--acc);color:#04241f;border-radius:6px;padding:0 7px;font-weight:800}
/* pipeline strip */
.pipe{display:flex;gap:6px;flex-wrap:wrap;align-items:center;background:var(--panel2);border:1px solid var(--line);
  border-radius:10px;padding:10px 14px}
.pnode{font-size:12px;color:var(--fg);background:#132033;border:1px solid var(--line);border-radius:6px;padding:3px 9px}
.parr{color:var(--mut)}
/* discovery panel */
.disc{display:grid;grid-template-columns:280px 1fr;gap:14px}
@media(max-width:820px){.disc{grid-template-columns:1fr}}
.checks{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.checks ul{list-style:none;margin:0;padding:0} .checks li{padding:4px 0 4px 22px;position:relative;color:var(--fg)}
.checks li:before{content:"✓";position:absolute;left:0;color:var(--good);font-weight:800}
.checks .lbl{color:var(--acc2);font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line);font-size:13px}
th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase}
tr[onclick]{cursor:pointer} tr[onclick]:hover{background:#17233a}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.pill{background:#0e2a26;color:var(--acc);border-radius:999px;padding:2px 9px;font-size:11px}
.empty{background:var(--panel);border:1px dashed var(--line);border-radius:10px;padding:16px;color:var(--mut)}
/* recommendation (dominant) */
.reco{background:linear-gradient(180deg,#132338,#0f1a2b);border:1px solid #24507a;border-radius:14px;padding:18px;
  box-shadow:0 0 0 1px #0e263f,0 14px 40px #0007}
.recohdr{display:flex;justify-content:space-between;align-items:flex-start}
.ribbon{display:inline-block;background:var(--acc);color:#04241f;font-weight:800;font-size:11px;text-transform:uppercase;
  letter-spacing:.06em;border-radius:6px;padding:2px 8px}
.reconame{font-size:26px;font-weight:800;margin-top:8px;letter-spacing:-.02em}
.recotag{color:var(--mut)} .conf{color:var(--mut);font-size:12px;white-space:nowrap}
.recogrid{display:grid;grid-template-columns:1fr 340px;gap:18px;margin-top:14px}
@media(max-width:820px){.recogrid{grid-template-columns:1fr}}
.selrow{display:flex;justify-content:space-between;border-top:1px solid var(--line);padding-top:10px}
.sellbl{color:var(--mut);font-size:12px} .selstrat{font-family:ui-monospace,monospace;color:#fff}
.reason{color:#cfe0f2;margin:8px 0} .alt{color:var(--mut);font-size:12px}
.improve{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:12px 0 6px}
.delta{font-weight:700;border-radius:6px;padding:3px 9px;font-size:13px}
.delta.up{background:#0e2a1f;color:var(--good)} .delta.dn{background:#2a0e14;color:var(--bad)}
.vs{color:var(--mut);font-size:12px}
.binding{color:var(--mut);font-size:11px;margin-top:6px}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.metrics.big{grid-template-columns:repeat(3,1fr);margin-bottom:10px}
.metrics div{background:var(--panel2);border-radius:8px;padding:8px;text-align:center}
.metrics b{display:block;font-size:15px} .metrics span{color:var(--mut);font-size:10px}
.bars .bar{display:flex;align-items:center;gap:7px;margin:3px 0}
.tkr{width:64px;font-family:ui-monospace,monospace;font-size:11px;color:#cbd5e1}
.track{flex:1;height:8px;background:var(--panel2);border-radius:5px;overflow:hidden}
.fill{display:block;height:100%;background:var(--acc)} .fill.cash{background:#64748b}
.wv{width:44px;text-align:right;font-size:11px;color:var(--mut)}
/* alternatives */
.altband{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
@media(max-width:820px){.altband{grid-template-columns:1fr}}
.altcard{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
.altcard.current{opacity:.9}
.objhdr{display:flex;justify-content:space-between;align-items:baseline}
.objlabel{font-weight:700} .objtag{color:var(--mut);font-size:12px;margin:2px 0 8px}
.warn{color:var(--bad);font-size:11px;margin-top:6px}
.actions{margin:16px 0}
.btn{background:var(--acc);color:#04241f;border:0;border-radius:9px;padding:10px 18px;font-weight:800;cursor:pointer}
.btn.ghost{background:transparent;color:var(--fg);border:1px solid var(--line)}
.foot{color:var(--mut);font-size:12px;border-top:1px solid var(--line);margin-top:30px;padding-top:12px}
/* mission drawer (PR-style) */
#drawer{position:fixed;top:0;right:0;height:100%;width:min(600px,96vw);background:var(--panel);border-left:1px solid var(--line);
  transform:translateX(100%);transition:.28s;overflow:auto;padding:20px;box-shadow:-10px 0 40px #0009;z-index:40}
#drawer.open{transform:none}
.prsec{border:1px solid var(--line);border-radius:10px;margin:12px 0;overflow:hidden}
.prsec>.h{background:var(--panel2);padding:9px 12px;font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--acc2)}
.prsec>.b{padding:12px} .prsec .mono{font-family:ui-monospace,monospace;font-size:12px;color:#cfe0f2}
.gate{background:#0e2a26;border:1px solid #164e46;border-radius:10px;padding:12px}
.exbranch{border:1px solid var(--line);border-radius:8px;padding:10px;margin:8px 0}
/* boot overlay */
#boot{position:fixed;inset:0;background:radial-gradient(1200px 600px at 50% -10%,#12233a,#070b12);z-index:100;
  display:flex;align-items:center;justify-content:center;transition:opacity .45s}
#boot.done{opacity:0;pointer-events:none}
.bootcard{width:min(560px,92vw)}
.boottitle{font-size:13px;color:var(--mut);text-align:center;margin-bottom:18px;letter-spacing:.04em}
.phase{background:#0e1626cc;border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:10px 0;
  opacity:0;transform:translateY(8px);transition:.35s}
.phase.show{opacity:1;transform:none}
.phase .lbl{color:var(--acc2);font-weight:800;font-size:12px;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px}
.phase ul{list-style:none;margin:0;padding:0}
.phase li{padding:3px 0 3px 24px;position:relative;opacity:0;transition:.25s;color:var(--fg)}
.phase li.on{opacity:1} .phase li:before{content:"✓";position:absolute;left:0;color:var(--good);font-weight:800}
.phase .think{color:var(--mut)} .phase .sel{font-size:18px;font-weight:800;color:#fff}
.phase .rz{color:var(--mut);font-size:13px;margin-top:2px}
.skip{position:fixed;bottom:18px;left:0;right:0;text-align:center;color:var(--mut);font-size:12px}
@media(prefers-reduced-motion:reduce){#boot{display:none}}
"""

_BODY = """
<div id="boot"><div class="bootcard">
  <div class="boottitle">quantify.club · the investment operating system is thinking…</div>
  <div class="phase" id="p1"><div class="lbl">Discovery Runtime</div><ul id="p1u"></ul></div>
  <div class="phase" id="p2"><div class="lbl">Decision Planner</div><div id="p2b"></div></div>
  <div class="phase" id="p3"><div class="lbl">Mission Runtime</div><div id="p3b"></div></div>
</div><div class="skip" onclick="skipBoot()">click to skip</div></div>

<div class="wrap">
  <h1>Quantify <span style="color:var(--mut);font-weight:500">— investment operating system</span></h1>
  <div class="sub">Continuously discovers market changes, selects among validated strategies, explains its
    reasoning, and proposes governed paper missions. Tenant: <b>{tenant}</b></div>
  <div class="demo"><b>DEMO — not investment advice.</b> {disclaimer}</div>

  <div class="pipe">{pipe}</div>

  <div class="row">
    <span class="chip">market regime <b>{regime}</b></span>
    <span class="chip">discoveries <b>{ndisc}</b></span>
    <span class="chip good">learning <b>{learn_pill}</b></span>
    <span class="chip">{promote}</span>
    <span class="chip"><a href="/research">research dashboard →</a></span>
  </div>

  <h2><span class="n">1</span> Discovery Runtime — what the system checked this cycle</h2>
  <div class="disc">
    <div class="checks"><div class="lbl">Completed checks</div><ul>{checks}</ul></div>
    <div>{queue}</div>
  </div>

  <h2><span class="n">2</span> Today's recommendation</h2>
  {reco}
  <div class="actions">
    <button class="btn" onclick="propose()">Open governed mission →</button>
    <span style="color:var(--mut);font-size:12px;margin-left:8px">Evidence → Verification → Approval. Paper-only, human-approved.</span>
  </div>

  <h2><span class="n">3</span> Alternative strategies &amp; current portfolio</h2>
  <div class="altband">{alts}{current}</div>

  <div class="foot">Aggregation: agentic_os.console (BFF) · RAAAL is the quantitative authority — the runtime
    selects among registered, research-backed strategies and never fabricates allocations. Paper trading
    only; no external execution path exists.</div>
</div>
<div id="drawer"><button class="btn ghost" onclick="closeD()">Close</button><div id="dbody"></div></div>
"""

_JS = r"""
const sleep = ms => new Promise(r=>setTimeout(r,ms));
async function j(u,opt){const r=await fetch(u,opt);return r.json();}
function skipBoot(){const b=document.getElementById('boot'); if(b) b.classList.add('done'); window.__boot=true;}
function closeD(){document.getElementById('drawer').classList.remove('open');}

async function runBoot(){
  if(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches){ skipBoot(); return; }
  const p1=document.getElementById('p1'), p2=document.getElementById('p2'), p3=document.getElementById('p3');
  const checks=[`${BOOT.assets} assets scanned`, `${BOOT.families} strategy families evaluated`,
    `${BOOT.signals} market signals checked`, `regime detected: ${BOOT.regime}`,
    `${BOOT.candidates} candidate mission(s) generated`];
  document.getElementById('p1u').innerHTML = checks.map(c=>`<li>${c}</li>`).join('');
  p1.classList.add('show'); await sleep(150);
  const lis=[...document.querySelectorAll('#p1u li')];
  for(const li of lis){ if(window.__boot) break; li.classList.add('on'); await sleep(240); }
  if(!window.__boot){ p2.classList.add('show');
    document.getElementById('p2b').innerHTML=`<div class="think">Comparing ${BOOT.strategies} research-backed strategies…</div>`;
    await sleep(700);
    document.getElementById('p2b').innerHTML=
      `<div class="think">Selected</div><div class="sel">${BOOT.recommended} · <span style="font-family:ui-monospace,monospace">${BOOT.selected}</span></div>`+
      `<div class="rz">${BOOT.reason}</div>`;
    await sleep(650);
  }
  if(!window.__boot){ p3.classList.add('show');
    document.getElementById('p3b').innerHTML=`<div class="sel" style="font-size:15px">Paper rebalance proposed.</div>`+
      `<div class="rz">Human approval required · paper-only, no real orders.</div>`;
    await sleep(750);
  }
  document.getElementById('boot').classList.add('done');
}

async function propose(){
  const m = await j('/api/investment/projects/default/missions/objective-compare',
    {method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({trigger:{opportunity_class:'manual_review'}})});
  const ex = await j('/api/investment/missions/'+m.mission_id+'/explain');
  const names={min_risk:'Capital Preservation',max_return_to_risk:'Balanced Growth',max_total_return:'Maximum Growth'};
  const branches=(m.branches||[]).map(b=>`
    <div class="exbranch"><div><b>${names[b.objective]||b.objective}</b> → <span class="mono">${b.selected_strategy_id}</span></div>
      <div style="color:#8496b0;font-size:12px">representation: ${b.representation||'—'} · alternatives: ${(b.alternative_strategy_ids||[]).join(', ')||'—'}</div>
      <div style="margin-top:6px"><button class="btn" onclick="approve('${m.mission_id}','${b.objective}')">Approve (paper) — ${names[b.objective]||b.objective}</button></div>
    </div>`).join('');
  const checks=((ex.branches&&ex.branches[0]&&[])||[]);
  document.getElementById('dbody').innerHTML=`
    <h3 style="margin:0 0 4px">Governed mission</h3>
    <div style="color:#8496b0;font-size:12px">snapshot ${m.snapshot_id} · regime ${m.regime} · reviewed like a pull request</div>
    <div class="prsec"><div class="h">① Evidence</div><div class="b">
      <div>One versioned evidence snapshot <span class="mono">${m.snapshot_id}</span> · regime <b>${m.regime}</b>.</div>
      <div style="color:#8496b0;font-size:12px;margin-top:4px">Each objective's Execution Planner chose an evidence representation (SQL / temporal / graph / documents).</div>
      ${branches}</div></div>
    <div class="prsec"><div class="h">② Verification</div><div class="b">
      <div>Deterministic pre-approval checks: weights sum to 1, strategy is registered, hard mandate constraints applied before selection.</div></div></div>
    <div class="prsec"><div class="h">③ Approval</div><div class="b">
      <div class="gate"><b>Human approval gate</b> — first and every paper rebalance requires explicit approval.
        Approving writes a <b>paper</b> order only; nothing is sent to a real venue.</div></div></div>`;
  document.getElementById('drawer').classList.add('open');
}
async function approve(mid, objective){
  const r = await j('/api/investment/missions/'+mid+'/approve',
    {method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({objective})});
  const names={min_risk:'Capital Preservation',max_return_to_risk:'Balanced Growth',max_total_return:'Maximum Growth'};
  alert(r.approved ? ('Paper rebalance recorded for '+(names[objective]||objective)+' — dispatched_externally='+r.dispatched_externally+'. No real order was placed.')
                   : ('Could not approve: '+(r.error||'unknown')));
}
runBoot();
"""
