"""Modern look & feel: CSS, colour system and small HTML components."""
from __future__ import annotations

import html
from datetime import datetime, timezone

import pandas as pd

# vivid, accessible palette (indigo / cyan / pink / emerald / amber / violet / rose / sky / lime / orange)
PALETTE = ["#6366F1", "#06B6D4", "#EC4899", "#10B981", "#F59E0B", "#8B5CF6", "#F43F5E", "#0EA5E9",
           "#84CC16", "#F97316", "#14B8A6", "#A855F7", "#EAB308", "#3B82F6", "#D946EF", "#22C55E"]

CATEGORY_COLORS = {
    "Technology": "#6366F1", "Business & Markets": "#10B981", "Startups & Funding": "#F59E0B",
    "Politics & Elections": "#F43F5E", "Geopolitics & Conflict": "#EF4444", "Health": "#06B6D4",
    "Climate & Disaster": "#0EA5E9", "Sports": "#84CC16", "Entertainment": "#EC4899",
    "Science & Space": "#8B5CF6", "Law & Justice": "#A16207", "Energy": "#F97316", "General": "#64748B",
}

STATUS_STYLE = {"🆕 Emerging": ("#F43F5E", "Emerging"), "📈 Trending": ("#F59E0B", "Trending"),
                "➖ Stable": ("#64748B", "Stable"), "📉 Fading": ("#0EA5E9", "Fading")}


def topic_color(i: int) -> str:
    return PALETTE[int(i) % len(PALETTE)] if i is not None and int(i) >= 0 else "#94A3B8"


def css(dark: bool) -> str:
    if dark:
        bg, panel, text, muted, border, side = "#0B1020", "rgba(30,41,59,.72)", "#E2E8F0", "#94A3B8", \
            "rgba(148,163,184,.18)", "#0F172A"
    else:
        bg, panel, text, muted, border, side = "#F6F7FB", "rgba(255,255,255,.92)", "#0F172A", "#64748B", \
            "rgba(15,23,42,.08)", "#FFFFFF"
    return f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {{
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important; }}
.stApp {{ background:
   radial-gradient(1200px 500px at 10% -10%, rgba(99,102,241,.18), transparent 60%),
   radial-gradient(900px 400px at 100% 0%, rgba(236,72,153,.12), transparent 60%), {bg}; color:{text}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.2rem; max-width: 1400px; }}
[data-testid="stSidebar"] > div:first-child {{ background:{side}; border-right:1px solid {border}; }}
[data-testid="stSidebar"] h1 {{ font-size:1.25rem; font-weight:800; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label, .stApp li, .stApp span {{ color:{text}; }}
small, .muted {{ color:{muted} !important; }}

/* ---------- hero ---------- */
.hero {{ border-radius:22px; padding:26px 30px; margin-bottom:18px; color:#fff;
  background: linear-gradient(120deg,#4F46E5 0%,#7C3AED 45%,#DB2777 100%);
  box-shadow: 0 20px 45px -20px rgba(79,70,229,.65); position:relative; overflow:hidden; }}
.hero:after {{ content:""; position:absolute; right:-80px; top:-80px; width:280px; height:280px;
  border-radius:50%; background:rgba(255,255,255,.10); }}
.hero h1 {{ color:#fff !important; font-size:2.0rem; font-weight:800; margin:0 0 6px 0; letter-spacing:-.02em; }}
.hero p {{ color:rgba(255,255,255,.88) !important; margin:0; font-size:.98rem; }}
.hero .meta {{ margin-top:14px; display:flex; gap:10px; flex-wrap:wrap; }}
.hero .tag {{ background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.25); color:#fff !important;
  padding:4px 12px; border-radius:999px; font-size:.8rem; font-weight:600; backdrop-filter: blur(6px); }}
.live {{ display:inline-flex; align-items:center; gap:7px; }}
.live i {{ width:9px; height:9px; border-radius:50%; background:#4ADE80; display:inline-block;
  box-shadow:0 0 0 0 rgba(74,222,128,.7); animation:pulse 1.6s infinite; }}
@keyframes pulse {{ 0% {{box-shadow:0 0 0 0 rgba(74,222,128,.7);}} 70% {{box-shadow:0 0 0 10px rgba(74,222,128,0);}}
  100% {{box-shadow:0 0 0 0 rgba(74,222,128,0);}} }}

/* ---------- KPI cards ---------- */
.kpis {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:14px; margin:4px 0 16px 0; }}
@media (max-width:1100px) {{ .kpis {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
.kpi {{ background:{panel}; border:1px solid {border}; border-radius:18px; padding:16px 18px;
  box-shadow:0 8px 24px -16px rgba(15,23,42,.35); transition:transform .15s ease, box-shadow .15s ease; }}
.kpi:hover {{ transform:translateY(-3px); box-shadow:0 14px 30px -16px rgba(79,70,229,.45); }}
.kpi .lbl {{ color:{muted} !important; font-size:.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; }}
.kpi .val {{ font-size:1.75rem; font-weight:800; margin-top:4px; color:{text} !important; }}
.kpi .sub {{ color:{muted} !important; font-size:.78rem; }}
.kpi .ic {{ float:right; font-size:1.4rem; opacity:.9; }}

/* ---------- cards / chips ---------- */
.card {{ background:{panel}; border:1px solid {border}; border-radius:18px; padding:16px 18px; margin-bottom:12px;
  box-shadow:0 8px 24px -18px rgba(15,23,42,.35); }}
.tcard {{ background:{panel}; border:1px solid {border}; border-radius:16px; padding:14px 16px; height:100%;
  border-top:4px solid var(--c); transition:transform .15s ease; }}
.tcard:hover {{ transform:translateY(-3px); }}
.tcard .t {{ font-weight:700; font-size:.98rem; color:{text} !important; }}
.tcard .n {{ color:{muted} !important; font-size:.8rem; margin:2px 0 8px 0; }}
.chip {{ display:inline-block; padding:3px 10px; margin:2px 4px 2px 0; border-radius:999px; font-size:.78rem;
  font-weight:600; background:color-mix(in srgb, var(--c) 14%, transparent); color:var(--c) !important;
  border:1px solid color-mix(in srgb, var(--c) 35%, transparent); }}
.badge {{ display:inline-block; padding:2px 9px; border-radius:8px; font-size:.72rem; font-weight:700;
  color:#fff !important; background:var(--c); }}
.bar {{ height:6px; border-radius:6px; background:{border}; overflow:hidden; margin-top:8px; }}
.bar > div {{ height:100%; background:var(--c); border-radius:6px; }}

/* ---------- article cards ---------- */
.art {{ display:flex; gap:12px; align-items:flex-start; background:{panel}; border:1px solid {border};
  border-left:4px solid var(--c); border-radius:14px; padding:11px 14px; margin-bottom:9px;
  transition:transform .12s ease, box-shadow .12s ease; }}
.art:hover {{ transform:translateX(3px); box-shadow:0 10px 24px -18px rgba(15,23,42,.5); }}
.art a {{ color:{text} !important; text-decoration:none; font-weight:600; }}
.art a:hover {{ text-decoration:underline; }}
.art .m {{ color:{muted} !important; font-size:.78rem; margin-top:3px; }}

/* ---------- alerts ---------- */
.alert {{ display:flex; gap:12px; align-items:center; background:linear-gradient(90deg,rgba(244,63,94,.12),
  rgba(245,158,11,.08)); border:1px solid rgba(244,63,94,.28); border-radius:14px; padding:10px 14px; margin:6px 0; }}
.alert b {{ color:#E11D48 !important; }}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap:6px; background:{panel}; padding:6px; border-radius:14px;
  border:1px solid {border}; }}
.stTabs [data-baseweb="tab"] {{ border-radius:10px; padding:8px 14px; height:auto; }}
.stTabs [aria-selected="true"] {{ background:linear-gradient(120deg,#6366F1,#8B5CF6); }}
.stTabs [aria-selected="true"] p {{ color:#fff !important; font-weight:700; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none; }}
div[data-testid="stPlotlyChart"] {{ background:{panel}; border:1px solid {border}; border-radius:18px; padding:6px; }}
.stButton > button, .stDownloadButton > button {{ border-radius:12px; font-weight:600; }}
.stButton > button[kind="primary"] {{ background:linear-gradient(120deg,#6366F1,#8B5CF6); border:0; }}
"""


# --------------------------------------------------------------------------- components
def esc(x) -> str:
    return html.escape(str(x or ""))


def ago(ts) -> str:
    try:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        secs = (datetime.now(timezone.utc) - ts.to_pydatetime()).total_seconds()
    except Exception:
        return ""
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    if secs < 86400:
        return f"{int(secs // 3600)} h ago"
    return f"{int(secs // 86400)} d ago"


def hero(title: str, subtitle: str, tags: list[str], live: bool) -> str:
    live_tag = "<span class='tag live'><i></i>LIVE</span>" if live else "<span class='tag'>STATIC</span>"
    t = "".join(f"<span class='tag'>{esc(x)}</span>" for x in tags)
    return f"<div class='hero'><h1>{esc(title)}</h1><p>{subtitle}</p><div class='meta'>{live_tag}{t}</div></div>"


def kpis(items: list[tuple[str, str, str, str]]) -> str:
    """items: (icon, label, value, sub)"""
    cards = "".join(f"<div class='kpi'><span class='ic'>{ic}</span><div class='lbl'>{esc(lbl)}</div>"
                    f"<div class='val'>{esc(val)}</div><div class='sub'>{esc(sub)}</div></div>"
                    for ic, lbl, val, sub in items)
    return f"<div class='kpis'>{cards}</div>"


def chip(text: str, color: str) -> str:
    return f"<span class='chip' style='--c:{color}'>{esc(text)}</span>"


def status_badge(status: str) -> str:
    color, label = STATUS_STYLE.get(status, ("#64748B", status))
    return f"<span class='badge' style='--c:{color}'>{esc(label)}</span>" if status else ""


def topic_card(tid: int, label: dict, size: int, share: float, words: list[str], status: str) -> str:
    c = topic_color(tid)
    chips = "".join(chip(w, c) for w in words[:5])
    return (f"<div class='tcard' style='--c:{c}'><div class='t'>{label['icon']} T{tid} · {esc(label['category'])}"
            f" {status_badge(status)}</div><div class='n'>{size} articles · {share:.1f}%</div>{chips}"
            f"<div class='bar'><div style='width:{min(100, share * 2.5):.0f}%'></div></div></div>")


def article_card(title: str, url: str, source: str, ts, topic_label: str, tid: int,
                 extra: str = "") -> str:
    c = topic_color(tid)
    link = f"<a href='{esc(url)}' target='_blank'>{esc(title)}</a>" if url else f"<b>{esc(title)}</b>"
    return (f"<div class='art' style='--c:{c}'><div style='flex:1'>{link}<div class='m'>"
            f"{chip(topic_label, c)} {esc(source)} · {ago(ts)} {extra}</div></div></div>")


def alert(msg: str) -> str:
    return f"<div class='alert'>🚨 <span><b>Alert</b> · {esc(msg)}</span></div>"
