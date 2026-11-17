"""Minimal stand-ins for `streamlit` and `plotly` so the dashboard can be
executed headlessly in CI (no browser, no server). The fake plotly validates
that every column referenced by a chart exists in its DataFrame."""
from __future__ import annotations

import contextlib
import types

import pandas as pd


# =========================================================================== plotly
class FakeFig:
    def __init__(self, kind, **kw):
        self.kind, self.kw, self.traces, self.layout = kind, kw, [], {}

    def update_layout(self, **kw):
        self.layout.update(kw)
        return self

    def update_traces(self, **kw):
        return self

    def update_xaxes(self, **kw):
        return self

    def update_yaxes(self, **kw):
        return self

    def add_trace(self, t):
        self.traces.append(t)
        return self


class Const:
    def __init__(self, v):
        self.v = v


def _px(kind):
    def fn(data_frame=None, *args, **kw):
        if isinstance(data_frame, pd.DataFrame):
            for k in ("x", "y", "color", "size", "text", "hover_name", "names", "values", "animation_frame"):
                v = kw.get(k)
                if isinstance(v, str):
                    assert v in data_frame.columns, f"px.{kind}: column {v!r} not in {list(data_frame.columns)}"
            for v in kw.get("path") or []:
                assert isinstance(v, Const) or v in data_frame.columns, f"px.{kind}: path column {v!r} missing"
            hd = kw.get("hover_data")
            for v in (hd.keys() if isinstance(hd, dict) else hd or []):
                assert v in data_frame.columns, f"px.{kind}: hover column {v!r} missing"
            assert len(data_frame) >= 0
        return FakeFig(kind, data=data_frame, **kw)

    return fn


def make_plotly() -> dict[str, types.ModuleType]:
    plotly = types.ModuleType("plotly")
    px = types.ModuleType("plotly.express")
    go = types.ModuleType("plotly.graph_objects")
    qual = types.SimpleNamespace(Pastel=["#a"] * 11, Set3=["#b"] * 12, Pastel2=["#c"] * 8, Dark24=["#d"] * 24)
    px.colors = types.SimpleNamespace(qualitative=qual)
    px.Constant = Const
    for k in ("bar", "pie", "line", "area", "scatter", "imshow", "treemap"):
        setattr(px, k, _px(k))
    go.Figure = lambda *a, **k: FakeFig("figure")
    go.Scatter = lambda **k: ("scatter", k)
    plotly.express, plotly.graph_objects = px, go
    return {"plotly": plotly, "plotly.express": px, "plotly.graph_objects": go}


# =========================================================================== streamlit
class SessionState(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


class FakeSt:
    """Records every call; widgets return their default (or an override)."""

    def __init__(self, root=None):
        self._root = root or self
        if root is None:
            self.calls: list[tuple] = []
            self.overrides: dict = {}
            self.session_state = SessionState()
            self.errors: list[str] = []
            self.reruns = 0

    # --- recording helpers
    def _rec(self, name, *a, **k):
        self._root.calls.append((name, a, k))

    def _ov(self, key, label, default):
        ov = self._root.overrides
        if key is not None and key in ov:
            return ov[key]
        if label in ov:
            return ov[label]
        return default

    # --- containers
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [FakeSt(self._root) for _ in range(n)]

    def tabs(self, names):
        self._rec("tabs", names)
        return [FakeSt(self._root) for _ in names]

    def expander(self, *a, **k):
        return FakeSt(self._root)

    def container(self, *a, **k):
        return FakeSt(self._root)

    def form(self, *a, **k):
        return FakeSt(self._root)

    def empty(self):
        return FakeSt(self._root)

    @contextlib.contextmanager
    def spinner(self, *a, **k):
        yield

    # --- widgets
    def text_input(self, label, value="", key=None, **k):
        return self._ov(key, label, value)

    def text_area(self, label, value="", key=None, **k):
        return self._ov(key, label, value)

    def checkbox(self, label, value=False, key=None, **k):
        return self._ov(key, label, value)

    toggle = checkbox

    def radio(self, label, options, index=0, key=None, **k):
        return self._ov(key, label, list(options)[index])

    def segmented_control(self, label, options, default=None, key=None, **k):
        return self._ov(key, label, default if default is not None else list(options)[0])

    def selectbox(self, label, options, index=0, key=None, **k):
        return self._ov(key, label, list(options)[index])

    def multiselect(self, label, options, default=None, key=None, **k):
        return self._ov(key, label, default or [])

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, key=None, **k):
        return self._ov(key, label, value if value is not None else min_value)

    def number_input(self, label, min_value=None, max_value=None, value=None, key=None, **k):
        return self._ov(key, label, value if value is not None else min_value)

    def file_uploader(self, label, key=None, **k):
        return self._ov(key, label, None)

    def button(self, label, key=None, **k):
        return self._ov(key, label, False)

    def form_submit_button(self, label="Submit", **k):
        return self._ov(None, label, False)

    def download_button(self, label, data, *a, **k):
        assert data is not None and len(data) > 0, f"empty download for {label}"
        self._rec("download_button", label)
        return False

    # --- output
    def error(self, msg, *a, **k):
        self._root.errors.append(str(msg))
        self._rec("error", msg)

    def plotly_chart(self, fig, key=None, **k):
        assert isinstance(fig, FakeFig), "plotly_chart expects a figure"
        assert ("use_container_width" in k) or ("width" in k)
        self._rec("plotly_chart", fig.kind, key)

    def dataframe(self, df, **k):
        assert isinstance(df, pd.DataFrame)
        self._rec("dataframe", df.shape)

    def rerun(self, *a, **k):
        self._root.reruns += 1

    def fragment(self, func=None, *, run_every=None):
        def deco(f):
            def wrapper(*a, **k):
                self._rec("fragment", run_every)
                return f(*a, **k)
            return wrapper
        return deco(func) if func else deco

    def __getattr__(self, name):  # title, caption, markdown, metric, info, json, ...
        if name.startswith("__"):
            raise AttributeError(name)

        def fn(*a, **k):
            self._rec(name, *a, **k)
            return None
        return fn

    # helpers for assertions
    def texts(self) -> str:
        return "\n".join(" ".join(str(x) for x in a) for n, a, k in self.calls)

    def count(self, name) -> int:
        return sum(1 for n, *_ in self.calls if n == name)


def make_streamlit(overrides: dict | None = None):
    st = FakeSt()
    st.overrides = dict(overrides or {})
    mod = types.ModuleType("streamlit")
    mod.__version__ = "1.40.0"
    mod._fake = st
    for name in dir(FakeSt):
        if not name.startswith("_"):
            setattr(mod, name, getattr(st, name))
    mod.sidebar = st          # sidebar shares the same recorder
    mod.session_state = st.session_state
    mod.column_config = types.SimpleNamespace(LinkColumn=lambda *a, **k: ("link", a),
                                              ProgressColumn=lambda *a, **k: ("progress", a))
    cfg = types.ModuleType("streamlit.config")
    _opts = {"theme.base": "light"}
    cfg.get_option = lambda k: _opts.get(k)
    cfg.set_option = lambda k, v: _opts.__setitem__(k, v)
    mod.config = cfg
    mod.__getattr__ = lambda name: getattr(st, name)
    return mod, {"streamlit": mod, "streamlit.config": cfg}
