"""NewsLens design system: newspaper typography (masthead + serif headlines),
BBC-style navigation, trending ticker, Google-News-style story clusters.
Pure CSS + small HTML builders; every user-visible string is escaped."""
from __future__ import annotations

import html
from datetime import datetime, timezone
from string import Template
from zoneinfo import ZoneInfo

import pandas as pd

BRAND = "NewsLens"
TAGLINE = "Every story, clustered in real time"

PALETTE = ["#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED", "#DB2777", "#0891B2", "#65A30D",
           "#EA580C", "#4F46E5", "#0D9488", "#9333EA", "#CA8A04", "#0284C7", "#C026D3", "#16A34A"]

CATEGORY_COLORS = {
    "Technology": "#2563EB", "Business & Markets": "#059669", "Startups & Funding": "#D97706",
    "Politics & Elections": "#C8102E", "Geopolitics & Conflict": "#B91C1C", "Health": "#0891B2",
    "Climate & Disaster": "#0284C7", "Sports": "#65A30D", "Entertainment": "#DB2777",
    "Science & Space": "#7C3AED", "Law & Justice": "#92400E", "Energy": "#EA580C", "General": "#475569",
}
SHORT_CATEGORY = {"Business & Markets": "Business", "Startups & Funding": "Startups",
                  "Politics & Elections": "Politics", "Geopolitics & Conflict": "World",
                  "Climate & Disaster": "Climate", "Science & Space": "Science", "Law & Justice": "Law",
                  "Sports": "Sport", "General": "News"}

STATUS_STYLE = {"🆕 Emerging": ("#C8102E", "Breaking"), "📈 Trending": ("#D97706", "Trending"),
                "➖ Stable": ("#64748B", "Developing"), "📉 Fading": ("#0284C7", "Earlier")}

REGION_TZ = {"IN": "Asia/Kolkata", "US": "America/New_York", "GB": "Europe/London",
             "AU": "Australia/Sydney", "CA": "America/Toronto", "SG": "Asia/Singapore"}
REGION_NAME = {"IN": "India", "US": "US", "GB": "UK", "AU": "Australia", "CA": "Canada", "SG": "Singapore"}


def topic_color(i) -> str:
    try:
        i = int(i)
    except Exception:
        return "#94A3B8"
    return PALETTE[i % len(PALETTE)] if i >= 0 else "#94A3B8"


def cat_color(cat: str) -> str:
    return CATEGORY_COLORS.get(cat, "#475569")


def short_cat(cat: str) -> str:
    return SHORT_CATEGORY.get(cat, cat)


# --------------------------------------------------------------------------- CSS
_CSS = Template(r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@700;800;900&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&display=swap');
:root { --nl-bg:$bg; --nl-panel:$panel; --nl-text:$text; --nl-muted:$muted; --nl-border:$border;
        --nl-red:#C8102E; --nl-nav:$nav; --nl-soft:$soft; }
html, body, .stApp, .stMarkdown, button, input, textarea, select, [class*="css"] {
  font-family:'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important; }
.stApp { background:var(--nl-bg); color:var(--nl-text); }
[data-testid="stHeader"] { background:transparent; height:0; }
/* left control panel */
[data-testid="stSidebar"] > div:first-child { background:var(--nl-panel); border-right:1px solid var(--nl-border); }
[data-testid="stSidebarUserContent"] { padding-top:1rem; }
.brand { display:flex; gap:12px; align-items:center; padding:2px 2px 14px 2px; border-bottom:3px double var(--nl-text);
  margin-bottom:4px; }
.brand .logo { width:40px; height:40px; display:flex; align-items:center; justify-content:center; color:#fff !important;
  font-family:'Playfair Display', Georgia, serif; font-weight:900; font-size:1.05rem; background:var(--nl-red); border-radius:4px; }
.brand .bn { font-family:'Playfair Display', Georgia, serif; font-weight:800; font-size:1.1rem; color:var(--nl-text) !important; line-height:1.2; }
.brand .bs { font-size:.74rem; color:var(--nl-muted) !important; letter-spacing:.04em; }
.sb-sec { font-size:.68rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; color:var(--nl-red) !important;
  margin:18px 0 4px 0; display:flex; align-items:center; gap:8px; }
.sb-sec:after { content:""; flex:1; height:1px; background:var(--nl-border); }
[data-testid="stSidebar"] label p, [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  font-size:.8rem !important; font-weight:600 !important; color:var(--nl-muted) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div, [data-testid="stSidebar"] input { border-radius:4px !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] { border:1px solid var(--nl-border); border-radius:4px; }
[data-testid="stSidebar"] [data-baseweb="tag"] { background:rgba(200,16,46,.10) !important; border-radius:3px; }
[data-testid="stSidebar"] [data-baseweb="tag"] span { color:var(--nl-red) !important; font-weight:600; }
[data-testid="stSidebar"] button[kind="segmented_controlActive"], [data-testid="stSidebar"] button[aria-checked="true"] {
  background:var(--nl-text) !important; color:var(--nl-bg) !important; border-color:var(--nl-text) !important; }
[data-testid="stSidebar"] button[kind="segmented_controlActive"] p { color:var(--nl-bg) !important; }
.sb-foot { margin-top:14px; padding-top:10px; border-top:1px solid var(--nl-border); font-size:.74rem;
  color:var(--nl-muted) !important; display:flex; align-items:center; gap:6px; }
.sb-foot .dot.ok { background:#16A34A; } .sb-foot .dot.lite { background:#D97706; }

/* research panel on reader pages */
.mstrip { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); border:1px solid var(--nl-border);
  border-left:4px solid var(--nl-red); background:var(--nl-soft); margin:2px 0 4px 0; }
@media (max-width:1100px) { .mstrip { grid-template-columns:repeat(3,minmax(0,1fr)); } }
.mstrip > div { padding:9px 14px; border-right:1px solid var(--nl-border); }
.mstrip > div:last-child { border-right:0; }
.mstrip .l { font-size:.66rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; color:var(--nl-muted) !important; }
.mstrip .v { font-family:'Source Serif 4', Georgia, serif; font-size:1.25rem; font-weight:700; color:var(--nl-text) !important; }
.mstrip .s { font-size:.7rem; color:var(--nl-muted) !important; }
.tline { font-size:.72rem; color:var(--nl-muted) !important; margin:2px 0 4px 0; }
.tline b { color:var(--nl-text) !important; font-weight:700; }
.tline .kw { display:inline-block; padding:0 6px; margin-right:4px; border-radius:3px; background:var(--nl-soft);
  border:1px solid var(--nl-border); color:var(--nl-text) !important; }
.block-container { padding-top:.6rem !important; max-width:1280px; }
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp p,.stApp label,.stApp li,.stApp span { color:var(--nl-text); }
a { color:inherit; }
.muted, small { color:var(--nl-muted) !important; }

/* top utility bar + masthead (The Hindu / Indian Express) */
.utility { display:flex; justify-content:space-between; align-items:center; font-size:.78rem;
  color:var(--nl-muted) !important; border-bottom:1px solid var(--nl-border); padding:4px 0 6px 0; }
.utility b { color:var(--nl-text) !important; }
.live { display:inline-flex; align-items:center; gap:6px; font-weight:700; color:var(--nl-red) !important;
  letter-spacing:.06em; }
.live i { width:8px; height:8px; border-radius:50%; background:var(--nl-red); display:inline-block;
  animation:nlpulse 1.5s infinite; }
@keyframes nlpulse { 0%{box-shadow:0 0 0 0 rgba(200,16,46,.6);} 70%{box-shadow:0 0 0 8px rgba(200,16,46,0);}
  100%{box-shadow:0 0 0 0 rgba(200,16,46,0);} }
.masthead { text-align:center; padding:14px 0 8px 0; }
.masthead .name { font-family:'Playfair Display', Georgia, serif !important; font-weight:900; font-size:3.1rem;
  letter-spacing:-.01em; line-height:1; color:var(--nl-text) !important; }
.masthead .name span { color:var(--nl-red) !important; }
.masthead .tag { font-size:.8rem; letter-spacing:.22em; text-transform:uppercase; color:var(--nl-muted) !important;
  margin-top:6px; }
.rule { border-top:3px double var(--nl-text); margin:6px 0 0 0; }

/* BBC-style navigation bar (segmented control restyled) */
.st-key-nav_main { background:var(--nl-nav); border-radius:0; padding:0 4px; margin-top:2px; }
.st-key-nav_main [data-testid="stWidgetLabel"] { display:none; }
.st-key-nav_main [data-testid="stButtonGroup"], .st-key-nav_main div[role="radiogroup"] { gap:0 !important; flex-wrap:wrap; }
.st-key-nav_main button { background:transparent !important; border:0 !important; border-radius:0 !important;
  color:#F1F5F9 !important; font-weight:600 !important; font-size:.9rem !important; padding:12px 14px !important;
  border-bottom:3px solid transparent !important; box-shadow:none !important; }
.st-key-nav_main button p { color:#F1F5F9 !important; }
.st-key-nav_main button:hover { background:rgba(255,255,255,.08) !important; }
.st-key-nav_main button[kind="segmented_controlActive"], .st-key-nav_main button[aria-checked="true"],
.st-key-nav_main button[aria-pressed="true"] { border-bottom:3px solid var(--nl-red) !important;
  background:rgba(255,255,255,.06) !important; }
.st-key-nav_more button, .st-key-nav_more [data-testid="stPopoverButton"] { background:var(--nl-nav) !important;
  color:#F1F5F9 !important; border:0 !important; border-radius:0 !important; font-weight:600; height:48px; width:100%; }
.st-key-nav_more p { color:#F1F5F9 !important; }
[data-testid="stPopoverBody"] .stButton button { width:100%; justify-content:flex-start; border:0; border-radius:8px;
  font-weight:600; background:transparent; }
[data-testid="stPopoverBody"] .stButton button:hover { background:var(--nl-soft); color:var(--nl-red); }

/* trending ticker */
.ticker { display:flex; align-items:stretch; border:1px solid var(--nl-border); background:var(--nl-panel);
  overflow:hidden; margin:10px 0 14px 0; height:40px; }
.ticker .lbl { background:var(--nl-red); color:#fff !important; font-weight:800; font-size:.75rem; letter-spacing:.14em;
  display:flex; align-items:center; padding:0 14px; white-space:nowrap; }
.ticker .track { flex:1; overflow:hidden; position:relative; display:flex; align-items:center; }
.ticker .move { display:inline-flex; gap:42px; white-space:nowrap; padding-left:100%;
  animation:nlscroll var(--dur,40s) linear infinite; font-size:.9rem; font-weight:500; }
.ticker:hover .move { animation-play-state:paused; }
.ticker .move b { color:var(--nl-red) !important; margin-right:6px; }
@keyframes nlscroll { from { transform:translateX(0); } to { transform:translateX(-100%); } }

/* section headers */
.sechead { display:flex; align-items:baseline; gap:12px; border-top:2px solid var(--nl-text); padding-top:8px;
  margin:18px 0 12px 0; }
.sechead h2 { font-family:'Source Serif 4', Georgia, serif !important; font-size:1.35rem !important; font-weight:700;
  margin:0 !important; padding:0 !important; }
.sechead span { color:var(--nl-muted) !important; font-size:.82rem; }
.pagehead { font-family:'Playfair Display', Georgia, serif !important; font-size:2.3rem; font-weight:800;
  border-bottom:4px solid var(--c, var(--nl-red)); padding-bottom:6px; margin:10px 0 4px 0; display:inline-block; }

/* stories */
.kicker { font-size:.72rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--c) !important; }
.kicker .st { margin-left:8px; padding:1px 7px; border-radius:3px; color:#fff !important; background:var(--s); font-size:.65rem; }
.hl { font-family:'Source Serif 4', Georgia, serif !important; font-weight:700; color:var(--nl-text) !important;
  text-decoration:none; line-height:1.22; display:block; }
.hl:hover { color:var(--nl-red) !important; }
.lead .hl { font-size:2rem; margin:6px 0 8px 0; }
.story .hl { font-size:1.12rem; margin:4px 0 6px 0; }
.summary { color:var(--nl-muted) !important; font-size:.95rem; line-height:1.55; margin:0 0 8px 0; }
.meta { color:var(--nl-muted) !important; font-size:.76rem; }
.meta b { color:var(--nl-text) !important; font-weight:600; }
.lead-img { width:100%; aspect-ratio:16/9; object-fit:cover; border-radius:4px; display:block; }
.ph { width:100%; aspect-ratio:16/9; border-radius:4px; display:flex; align-items:flex-end; padding:16px;
  background:linear-gradient(135deg, var(--c), color-mix(in srgb, var(--c) 40%, #0F172A)); color:#fff !important;
  font-family:'Playfair Display', Georgia, serif; font-size:1.6rem; font-weight:800; }
.ph.sm { aspect-ratio:auto; height:84px; font-size:1rem; padding:10px; }
.thumb { width:110px; height:74px; object-fit:cover; border-radius:4px; flex-shrink:0; }
.story { border-top:1px solid var(--nl-border); padding:12px 0 6px 0; }
.story .row { display:flex; gap:14px; }
.related { list-style:none; padding:0; margin:4px 0 2px 0; }
.related li { font-size:.86rem; padding:5px 0 5px 12px; border-left:2px solid var(--nl-border); margin-bottom:3px; }
.related li a { text-decoration:none; color:var(--nl-text) !important; font-weight:500; }
.related li a:hover { color:var(--nl-red) !important; }
.related li .src { color:var(--nl-muted) !important; font-size:.75rem; margin-left:6px; }
.chip { display:inline-block; padding:2px 9px; margin:2px 4px 2px 0; border-radius:3px; font-size:.75rem; font-weight:600;
  background:color-mix(in srgb, var(--c) 10%, transparent); color:var(--c) !important;
  border:1px solid color-mix(in srgb, var(--c) 30%, transparent); }

/* "Full coverage" buttons rendered as text links */
[class*="st-key-fc_"] button { background:transparent !important; border:0 !important; padding:0 !important;
  min-height:0 !important; color:var(--nl-red) !important; box-shadow:none !important; }
[class*="st-key-fc_"] button p { color:var(--nl-red) !important; font-weight:700 !important; font-size:.8rem !important; }
[class*="st-key-fc_"] button:hover p { text-decoration:underline; }
[class*="st-key-back"] button { border-radius:20px; font-weight:600; }

/* right rail (Most read / Latest) */
.rail { background:var(--nl-panel); border:1px solid var(--nl-border); padding:12px 14px; margin-bottom:14px; }
.rail h4 { font-family:'Source Serif 4', Georgia, serif !important; font-size:1.05rem !important; margin:0 0 8px 0 !important;
  padding:0 0 6px 0 !important; border-bottom:2px solid var(--nl-red); }
.rank { display:flex; gap:12px; padding:8px 0; border-bottom:1px solid var(--nl-border); }
.rank:last-child { border-bottom:0; }
.rank .n { font-family:'Playfair Display', Georgia, serif; font-size:1.7rem; font-weight:800; color:var(--nl-red) !important;
  line-height:1; min-width:26px; }
.rank .t { font-weight:600; font-size:.88rem; line-height:1.3; }
.latest { display:flex; gap:10px; padding:7px 0; border-bottom:1px solid var(--nl-border); font-size:.86rem; }
.latest:last-child { border-bottom:0; }
.latest .tm { color:var(--nl-red) !important; font-weight:700; font-size:.75rem; min-width:46px; padding-top:2px; }
.latest a { text-decoration:none; color:var(--nl-text) !important; font-weight:500; }
.latest a:hover { color:var(--nl-red) !important; }

/* insights (analytics) */
.kpis { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin:6px 0 14px 0; }
@media (max-width:1000px) { .kpis { grid-template-columns:repeat(2,minmax(0,1fr)); } }
.kpi { background:var(--nl-panel); border:1px solid var(--nl-border); border-top:3px solid var(--nl-red); padding:12px 14px; }
.kpi .lbl { color:var(--nl-muted) !important; font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
.kpi .val { font-family:'Source Serif 4', Georgia, serif; font-size:1.7rem; font-weight:700; color:var(--nl-text) !important; }
.kpi .sub { color:var(--nl-muted) !important; font-size:.74rem; }
.tcard { background:var(--nl-panel); border:1px solid var(--nl-border); border-top:3px solid var(--c); padding:12px 14px; height:100%; }
.tcard .t { font-weight:700; font-size:.95rem; color:var(--nl-text) !important; }
.tcard .n { color:var(--nl-muted) !important; font-size:.78rem; margin:2px 0 8px 0; }
.badge { display:inline-block; padding:1px 7px; border-radius:3px; font-size:.66rem; font-weight:700; color:#fff !important;
  background:var(--c); text-transform:uppercase; letter-spacing:.05em; vertical-align:middle; }
.bar { height:4px; background:var(--nl-border); margin-top:8px; } .bar > div { height:100%; background:var(--c); }
.card { background:var(--nl-panel); border:1px solid var(--nl-border); padding:14px 16px; margin-bottom:12px; }
.art { display:flex; gap:12px; background:var(--nl-panel); border:1px solid var(--nl-border); border-left:3px solid var(--c);
  padding:10px 14px; margin-bottom:8px; }
.art a { color:var(--nl-text) !important; text-decoration:none; font-weight:600; }
.art a:hover { color:var(--nl-red) !important; }
.art .m { color:var(--nl-muted) !important; font-size:.76rem; margin-top:3px; }
.stTabs [data-baseweb="tab-list"] { gap:2px; border-bottom:1px solid var(--nl-border); }
.stTabs [data-baseweb="tab"] { padding:10px 14px; }
.stTabs [aria-selected="true"] p { color:var(--nl-red) !important; font-weight:700; }
.stTabs [data-baseweb="tab-highlight"] { background:var(--nl-red); }
div[data-testid="stPlotlyChart"] { background:var(--nl-panel); border:1px solid var(--nl-border); padding:4px; }
.stButton > button[kind="primary"] { background:var(--nl-red); border-color:var(--nl-red); }

/* footer */
.foot { border-top:3px double var(--nl-text); margin-top:30px; padding:12px 0 30px 0; display:flex;
  justify-content:space-between; flex-wrap:wrap; gap:8px; font-size:.78rem; color:var(--nl-muted) !important; }
.foot .b { font-family:'Playfair Display', Georgia, serif; font-weight:800; font-size:1.1rem; color:var(--nl-text) !important; }
.dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:4px; }
""")


def css(dark: bool) -> str:
    if dark:
        v = dict(bg="#0E1117", panel="#161B22", text="#E6EDF3", muted="#9DA7B3", border="#2A313C",
                 nav="#000000", soft="#1F2630")
    else:
        v = dict(bg="#FFFFFF", panel="#FFFFFF", text="#121212", muted="#5F6368", border="#E3E3E3",
                 nav="#121212", soft="#F3F4F6")
    return _CSS.substitute(v)


# --------------------------------------------------------------------------- helpers
def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def _ts(ts) -> pd.Timestamp | None:
    try:
        t = pd.Timestamp(ts)
        return t.tz_localize("UTC") if t.tzinfo is None else t
    except Exception:
        return None


def ago(ts) -> str:
    t = _ts(ts)
    if t is None:
        return ""
    secs = (datetime.now(timezone.utc) - t.to_pydatetime()).total_seconds()
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    if secs < 86400:
        return f"{int(secs // 3600)} h ago"
    return f"{int(secs // 86400)} d ago"


def clock(ts, region: str = "IN") -> str:
    t = _ts(ts)
    if t is None:
        return ""
    try:
        return t.tz_convert(ZoneInfo(REGION_TZ.get(region, "UTC"))).strftime("%H:%M")
    except Exception:
        return t.strftime("%H:%M")


def link(title: str, url: str, cls: str = "hl") -> str:
    if url:
        return f"<a class='{cls}' href='{esc(url)}' target='_blank' rel='noopener'>{esc(title)}</a>"
    return f"<span class='{cls}'>{esc(title)}</span>"


def kicker(category: str, status: str = "") -> str:
    st_html = ""
    if status in STATUS_STYLE and status != "➖ Stable":
        c, lbl = STATUS_STYLE[status]
        st_html = f"<span class='st' style='--s:{c}'>{lbl}</span>"
    return f"<div class='kicker' style='--c:{cat_color(category)}'>{esc(short_cat(category))}{st_html}</div>"


def chip(text: str, color: str) -> str:
    return f"<span class='chip' style='--c:{color}'>{esc(text)}</span>"


def status_badge(status: str) -> str:
    if not status:
        return ""
    color, label = STATUS_STYLE.get(status, ("#64748B", status))
    return f"<span class='badge' style='--c:{color}'>{esc(label)}</span>"


def image_or_placeholder(url: str, category: str, small: bool = False) -> str:
    if url:
        cls = "thumb" if small else "lead-img"
        return f"<img class='{cls}' src='{esc(url)}' loading='lazy' onerror=\"this.style.display='none'\">"
    if small:
        return ""
    return f"<div class='ph' style='--c:{cat_color(category)}'>{esc(short_cat(category))}</div>"


# --------------------------------------------------------------------------- page chrome
def utility_bar(region: str, live: bool, updated: str) -> str:
    try:
        now = datetime.now(ZoneInfo(REGION_TZ.get(region, "UTC")))
    except Exception:
        now = datetime.now(timezone.utc)
    left = f"<span><b>{now.strftime('%A, %d %B %Y')}</b> · {REGION_NAME.get(region, region)} edition</span>"
    state = "<span class='live'><i></i>LIVE</span>" if live else "<span class='muted'>Offline sample edition</span>"
    return f"<div class='utility'>{left}<span>{state} · Updated {esc(updated)}</span></div>"


def masthead() -> str:
    return (f"<div class='masthead'><div class='name'>News<span>Lens</span></div>"
            f"<div class='tag'>{esc(TAGLINE)}</div></div><div class='rule'></div>")


def ticker(items: list[tuple[str, str]]) -> str:
    """items: (label, text)"""
    if not items:
        return ""
    body = "".join(f"<span><b>{esc(a)}</b>{esc(b)}</span>" for a, b in items)
    dur = max(25, 9 * len(items))
    return (f"<div class='ticker'><div class='lbl'>TRENDING</div><div class='track'>"
            f"<div class='move' style='--dur:{dur}s'>{body}</div></div></div>")


def section_head(title: str, sub: str = "") -> str:
    return f"<div class='sechead'><h2>{esc(title)}</h2><span>{esc(sub)}</span></div>"


def page_head(title: str, color: str = "#C8102E") -> str:
    return f"<div class='pagehead' style='--c:{color}'>{esc(title)}</div>"


def footer(status_html: str) -> str:
    return (f"<div class='foot'><span class='b'>News<span style='color:#C8102E'>Lens</span></span>"
            f"<span>{status_html}</span><span>Topic modeling with BERTopic &amp; LDA · headlines link to "
            f"the original publishers</span></div>")


# --------------------------------------------------------------------------- research panel
def model_strip(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(f"<div><div class='l'>{esc(a)}</div><div class='v'>{esc(b)}</div><div class='s'>{esc(c)}</div></div>"
                    for a, b, c in items)
    return f"<div class='mstrip'>{cells}</div>"


def topic_line(c: dict, n: int = 4) -> str:
    kws = "".join(f"<span class='kw'>{esc(w)}</span>" for w in c.get("keywords", [])[:n])
    return f"<div class='tline'><b>Topic {c['tid']}</b> · {kws}</div>"


# --------------------------------------------------------------------------- stories
def related_list(rows: list[dict]) -> str:
    if not rows:
        return ""
    items = "".join(f"<li>{link(r['title'], r.get('url', ''), '')}<span class='src'>{esc(r['source'])}</span></li>"
                    for r in rows)
    return f"<ul class='related'>{items}</ul>"


def lead_story(c: dict) -> str:
    lead = c["lead"]
    meta = (f"<div class='meta'><b>{esc(lead['source'])}</b> · {ago(lead['published_at'])} · "
            f"{c['size']} articles from {c['n_sources']} sources</div>")
    summary = f"<p class='summary'>{esc(_trim(lead.get('description', ''), 260))}</p>" if lead.get("description") else ""
    return (f"<div class='lead'>{image_or_placeholder(lead.get('image', ''), c['category'])}"
            f"<div style='margin-top:10px'>{kicker(c['category'], c['status'])}"
            f"{link(lead['title'], lead.get('url', ''))}{summary}{meta}{topic_line(c, 6)}"
            f"{related_list(c['related'][:3])}</div></div>")


def story_card(c: dict, n_related: int = 2) -> str:
    lead = c["lead"]
    img = image_or_placeholder(lead.get("image", ""), c["category"], small=True)
    meta = (f"<div class='meta'><b>{esc(lead['source'])}</b> · {ago(lead['published_at'])} · "
            f"{c['size']} articles</div>")
    return (f"<div class='story'><div class='row'><div style='flex:1'>{kicker(c['category'], c['status'])}"
            f"{link(lead['title'], lead.get('url', ''))}{meta}{topic_line(c)}</div>{img}</div>"
            f"{related_list(c['related'][:n_related])}</div>")


def rail_ranked(title: str, clusters: list[dict]) -> str:
    rows = "".join(f"<div class='rank'><div class='n'>{i}</div><div><div class='t'>{esc(_trim(c['lead']['title'], 90))}"
                   f"</div><div class='meta'>{esc(short_cat(c['category']))} · {c['size']} articles"
                   f"{' · ' + STATUS_STYLE[c['status']][1] if c['status'] in STATUS_STYLE else ''}</div></div></div>"
                   for i, c in enumerate(clusters, 1))
    return f"<div class='rail'><h4>{esc(title)}</h4>{rows}</div>"


def rail_latest(title: str, rows: list[dict], region: str) -> str:
    items = "".join(f"<div class='latest'><div class='tm'>{clock(r['published_at'], region)}</div><div>"
                    f"{link(_trim(r['title'], 110), r.get('url', ''), '')}<div class='meta'>{esc(r['source'])}</div>"
                    f"</div></div>" for r in rows)
    return f"<div class='rail'><h4>{esc(title)}</h4>{items}</div>"


def _trim(s: str, n: int) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + "…"


# --------------------------------------------------------------------------- insights components
def kpis(items: list[tuple[str, str, str, str]]) -> str:
    cards = "".join(f"<div class='kpi'><div class='lbl'>{esc(lbl)}</div><div class='val'>{esc(val)}</div>"
                    f"<div class='sub'>{esc(sub)}</div></div>" for _ic, lbl, val, sub in items)
    return f"<div class='kpis'>{cards}</div>"


def topic_card(tid: int, label: dict, size: int, share: float, words: list[str], status: str) -> str:
    c = topic_color(tid)
    chips = "".join(chip(w, c) for w in words[:5])
    return (f"<div class='tcard' style='--c:{c}'><div class='t'>T{tid} · {esc(short_cat(label['category']))} "
            f"{status_badge(status)}</div><div class='n'>{size} articles · {share:.1f}%</div>{chips}"
            f"<div class='bar'><div style='width:{min(100, share * 2.5):.0f}%'></div></div></div>")


def article_card(title: str, url: str, source: str, ts, topic_label: str, tid: int, extra: str = "") -> str:
    c = topic_color(tid)
    return (f"<div class='art' style='--c:{c}'><div style='flex:1'>{link(title, url, '')}<div class='m'>"
            f"{chip(topic_label, c)} {esc(source)} · {ago(ts)} {esc(extra)}</div></div></div>")
