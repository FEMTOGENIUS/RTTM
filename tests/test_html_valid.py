from html.parser import HTMLParser
from tests.test_dashboard_smoke import run_app, BASE

class P(HTMLParser):
    VOID = {"br", "img", "hr", "input", "meta", "link"}
    def __init__(s):
        super().__init__(); s.stack = []; s.err = []
    def handle_starttag(s, t, a):
        if t not in s.VOID: s.stack.append(t)
    def handle_endtag(s, t):
        if s.stack and s.stack[-1] == t: s.stack.pop()
        else: s.err.append(t)

def test_all_html_balanced(monkeypatch):
    for sess in ({}, {"page": "Business"}, {"page": "topic:1"}):
        st, _ = run_app(monkeypatch, BASE, sess)
        for n, a, k in st.calls:
            if n == "markdown" and k.get("unsafe_allow_html") and not str(a[0]).startswith("<style>"):
                p = P(); p.feed(a[0]); assert not p.stack and not p.err, (a[0][:200], p.stack, p.err)
