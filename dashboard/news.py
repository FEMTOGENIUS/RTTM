"""NewsLens reader experience: Google-News-style story clusters, newspaper
section pages and "Full coverage" pages - all driven by the topic model."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts, theme

# BBC-style top navigation: visible sections + "More" dropdown
NAV_MAIN = ["Home", "Politics", "World", "Business", "Tech", "Sport", "Health", "Entertainment", "Insights"]
SECTIONS: dict[str, list[str]] = {
    "Politics": ["Politics & Elections"],
    "World": ["Geopolitics & Conflict"],
    "Business": ["Business & Markets", "Startups & Funding", "Energy"],
    "Tech": ["Technology", "Science & Space"],
    "Sport": ["Sports"],
    "Health": ["Health"],
    "Entertainment": ["Entertainment"],
    # "More" dropdown
    "Climate": ["Climate & Disaster"],
    "Law & Justice": ["Law & Justice"],
    "Science & Space": ["Science & Space"],
    "Startups": ["Startups & Funding"],
    "Energy": ["Energy"],
    "Other news": ["General"],
}
MORE = ["Climate", "Law & Justice", "Science & Space", "Startups", "Energy", "Other news"]


# --------------------------------------------------------------------------- navigation state
def go(page: str) -> None:
    """Button / menu callback - runs before widgets are drawn, so it may set widget state."""
    st.session_state["page"] = page
    st.session_state["nav_main"] = page if page in NAV_MAIN else None


def _nav_changed() -> None:
    v = st.session_state.get("nav_main")
    if v:
        st.session_state["page"] = v


def current_page() -> str:
    return st.session_state.get("page", "Home")


def render_nav() -> None:
    page = current_page()
    c1, c2 = st.columns([12, 1.6], gap="small")
    with c1:
        kw = {} if "nav_main" in st.session_state else {"default": page if page in NAV_MAIN else None}
        fn = getattr(st, "segmented_control", None)
        if callable(fn):
            with st.container(key="nav_main_wrap"):
                fn("Sections", NAV_MAIN, key="nav_main", on_change=_nav_changed, label_visibility="collapsed", **kw)
        else:  # very old Streamlit
            choice = st.radio("Sections", NAV_MAIN, horizontal=True, label_visibility="collapsed")
            st.session_state["page"] = choice
    with c2:
        with st.container(key="nav_more"):
            with _popover("More ▾"):
                for name in MORE:
                    st.button(name, key=f"more_{name}", on_click=go, args=(name,), **_wide())


def _popover(label: str):
    pop = getattr(st, "popover", None)
    if callable(pop):
        try:
            return pop(label, use_container_width=True)
        except TypeError:
            return pop(label)
    return st.expander(label)


def _wide() -> dict:
    return _stretch_kw()


def fc_button(c: dict, where: str) -> None:
    st.button(f"Full coverage · {c['size']} articles →", key=f"fc_{where}_{c['tid']}",
              on_click=go, args=(f"topic:{c['tid']}",))


# --------------------------------------------------------------------------- clusters
def build_clusters(out, model: str) -> list[dict]:
    """Turn topics into news stories: lead article, related coverage, status, score."""
    if model not in out.results:
        return []
    res, lab = out.results[model], out.labels[model]
    docs = out.doc_frame(model)
    if "image" not in docs:
        docs["image"] = ""
    docs["image"] = docs["image"].fillna("")
    tr = out.trends.get(model, pd.DataFrame())
    tr = tr.set_index("topic") if not tr.empty else tr
    clusters = []
    for t in res.topic_ids:
        d = docs[docs["topic"] == t]
        if d.empty:
            continue
        d = d.sort_values(["confidence", "published_at"], ascending=[False, False])
        top = d.head(4)
        with_img = top[top["image"].astype(bool)]
        lead = (with_img.iloc[0] if len(with_img) else d.iloc[0]).to_dict()
        related, seen = [], {lead["source"]}
        rest = d[d["id"] != lead["id"]].sort_values("published_at", ascending=False)
        for _, r in rest.iterrows():                 # prefer other publishers first
            if r["source"] not in seen:
                related.append(r.to_dict())
                seen.add(r["source"])
        for _, r in rest.iterrows():
            if len(related) >= 8:
                break
            if all(r["id"] != x["id"] for x in related):
                related.append(r.to_dict())
        status = tr.loc[t, "status"] if t in getattr(tr, "index", []) else ""
        recent = int(tr.loc[t, "recent"]) if t in getattr(tr, "index", []) else 0
        score = len(d) + 1.5 * recent + (8 if str(status).endswith("Emerging") else 0) \
            + (4 if str(status).endswith("Trending") else 0)
        clusters.append(dict(tid=int(t), label=lab[t], category=lab[t]["category"], size=len(d),
                             n_sources=int(d["source"].nunique()), lead=lead, related=related, status=status,
                             recent=recent, score=score, keywords=res.top_words(t, 8), docs=d))
    return sorted(clusters, key=lambda c: -c["score"])


def ticker_items(clusters: list[dict]) -> list[tuple[str, str]]:
    hot = [c for c in clusters if str(c["status"]).endswith(("Emerging", "Trending"))] or clusters
    return [(theme.short_cat(c["category"]).upper(), theme._trim(c["lead"]["title"], 95)) for c in hot[:8]]


# --------------------------------------------------------------------------- pages
def _story_grid(clusters: list[dict], where: str, cols: int = 2) -> None:
    for i in range(0, len(clusters), cols):
        columns = st.columns(cols, gap="large")
        for col, c in zip(columns, clusters[i:i + cols]):
            with col:
                st.markdown(theme.story_card(c), unsafe_allow_html=True)
                fc_button(c, where)


def _rail(clusters: list[dict], docs: pd.DataFrame, region: str, word_freq=None) -> None:
    st.markdown(theme.rail_ranked("Most covered", sorted(clusters, key=lambda c: -c["size"])[:6]),
                unsafe_allow_html=True)
    latest = docs.sort_values("published_at", ascending=False).head(10).to_dict("records")
    st.markdown(theme.rail_latest("Latest", latest, region), unsafe_allow_html=True)
    if word_freq:
        chips = "".join(theme.chip(w, "#C8102E") for w, _ in word_freq[:14])
        st.markdown(f"<div class='rail'><h4>Hot keywords</h4>{chips}</div>", unsafe_allow_html=True)


def render_home(out, model: str, clusters: list[dict], region: str) -> None:
    if not clusters:
        st.info("No stories yet.")
        return
    docs = out.doc_frame(model)
    main, rail = st.columns([2.15, 1], gap="large")
    with main:
        lead = clusters[0]
        st.markdown(theme.lead_story(lead), unsafe_allow_html=True)
        fc_button(lead, "lead")
        st.markdown(theme.section_head("Top stories", f"{len(clusters)} stories developing now"),
                    unsafe_allow_html=True)
        _story_grid(clusters[1:11], "top")
    with rail:
        _rail(clusters, docs, region, out.word_freq)
    shown = {c["tid"] for c in clusters[:11]}
    for name in NAV_MAIN[1:-1]:      # newspaper-style section blocks
        sec = [c for c in clusters if c["category"] in SECTIONS[name] and c["tid"] not in shown]
        if not sec:
            continue
        st.markdown(theme.section_head(name, f"{len(sec)} more"), unsafe_allow_html=True)
        _story_grid(sec[:3], f"sec_{name}", cols=3)
        st.button(f"More {name} →", key=f"fc_more_{name}", on_click=go, args=(name,))


def render_section(out, model: str, clusters: list[dict], name: str, region: str) -> None:
    cats = SECTIONS.get(name, [])
    color = theme.cat_color(cats[0]) if cats else "#C8102E"
    st.markdown(theme.page_head(name, color), unsafe_allow_html=True)
    sec = [c for c in clusters if c["category"] in cats]
    if not sec:
        st.info(f"No {name} stories in the current news window. Try a wider time window or another search.")
        return
    docs = pd.concat([c["docs"] for c in sec])
    main, rail = st.columns([2.15, 1], gap="large")
    with main:
        st.markdown(theme.lead_story(sec[0]), unsafe_allow_html=True)
        fc_button(sec[0], "s_lead")
        if len(sec) > 1:
            st.markdown(theme.section_head(f"More in {name}"), unsafe_allow_html=True)
            _story_grid(sec[1:], f"s_{name}")
    with rail:
        _rail(sec, docs, region)


def render_topic(out, model: str, clusters: list[dict], tid: int, region: str, dark: bool) -> None:
    c = next((x for x in clusters if x["tid"] == tid), None)
    st.button("← Back to Home", key="back_home", on_click=go, args=("Home",))
    if c is None:
        st.info("This story is no longer in the current news window.")
        return
    lead = c["lead"]
    color = theme.cat_color(c["category"])
    st.markdown(f"{theme.kicker(c['category'], c['status'])}"
                f"<h1 class='hl' style='font-size:2.3rem;margin:6px 0 10px 0'>{theme.esc(lead['title'])}</h1>"
                f"<div class='meta'>Full coverage · <b>{c['size']}</b> articles from <b>{c['n_sources']}</b> "
                f"sources · latest {theme.ago(c['docs']['published_at'].max())}</div>"
                f"<div style='margin:10px 0'>{''.join(theme.chip(w, color) for w in c['keywords'])}</div>",
                unsafe_allow_html=True)
    main, rail = st.columns([2.15, 1], gap="large")
    with main:
        if lead.get("description"):
            st.markdown(f"<p class='summary' style='font-size:1.05rem'>{theme.esc(lead['description'])}</p>",
                        unsafe_allow_html=True)
        st.markdown(theme.section_head("All coverage", "newest first"), unsafe_allow_html=True)
        rows = c["docs"].sort_values("published_at", ascending=False).to_dict("records")
        st.markdown("".join(
            f"<div class='latest'><div class='tm'>{theme.clock(r['published_at'], region)}</div><div>"
            f"{theme.link(r['title'], r.get('url', ''), '')}<div class='meta'><b>{theme.esc(r['source'])}</b> · "
            f"{theme.ago(r['published_at'])}</div></div></div>" for r in rows), unsafe_allow_html=True)
    with rail:
        names = {c["tid"]: c["label"]["name"]}
        from utils.trends import topics_over_time

        tl = topics_over_time(c["docs"], c["docs"]["topic"].to_numpy(), names)
        if not tl.empty and tl["time"].nunique() > 1:
            fig = charts.timeline(tl, "count", dark)
            fig.update_layout(height=260, showlegend=False, title=None)
            st.plotly_chart(fig, key=f"tl_{tid}", **_stretch_kw())
        src = c["docs"]["source"].value_counts().head(8)
        st.markdown("<div class='rail'><h4>Who is covering it</h4>" + "".join(
            f"<div class='latest'><div class='tm'>{n}</div><div>{theme.esc(s)}</div></div>" for s, n in src.items())
            + "</div>", unsafe_allow_html=True)
    others = [x for x in clusters if x["category"] == c["category"] and x["tid"] != tid][:4]
    if others:
        st.markdown(theme.section_head(f"More in {theme.short_cat(c['category'])}"), unsafe_allow_html=True)
        _story_grid(others, "rel")


def _stretch_kw() -> dict:
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
        if (major, minor) >= (1, 50):
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}
