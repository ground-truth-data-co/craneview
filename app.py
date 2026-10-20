"""CraneView — eyeball your CSV for data-entry anomalies.

Upload a CSV (or pick a bundled sample). The summary table lists every
variable with its data type; the type is a dropdown, so you can
override the detection, and the checkbox beside each variable decides
whether it appears in the plot. The plot stacks the selected variables
in panels over a shared row-index axis — value vs. row order is what
makes entry anomalies (a stray 999, a block in the wrong unit, swapped
rows) visible.

Privacy: uploaded files are processed in memory for this session only.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from craneview_core import KINDS, analyze, as_datetime, format_details

# Qualitative palette for categories: Okabe-Ito, designed so that
# neighboring categories stay distinguishable (and colorblind-safe).
# Viridis is a sequential palette — right for continuous values, wrong
# for categories, whose shades blur into each other. Beyond these,
# append Plotly's Safe qualitative set, then cycle.
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
CATEGORY_COLORS = OKABE_ITO + [c for c in plotly.colors.qualitative.Safe if c not in OKABE_ITO]

# Above this many rows, plots draw a stride sample so the browser stays
# responsive. Summaries always use the full data; only the dots thin out.
MAX_PLOT_POINTS = 20_000

SAMPLES = {
    "Messy plants (100 rows, planted errors)": "messy_plants.csv",
    "Iris (150 rows)": "iris.csv",
    "Cars — mpg (398 rows)": "mpg.csv",
}
SAMPLES_DIR = Path(__file__).parent / "samples"

st.set_page_config(page_title="CraneView", page_icon="🪶", layout="wide")
st.title("CraneView")


def read_csv(file) -> pd.DataFrame:
    """Read an uploaded CSV, falling back to latin-1 for older exports."""
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="latin-1")


uploaded = st.file_uploader("Upload a CSV", type=["csv"])
sample_choice = st.selectbox("Or try a sample dataset", ["(none)", *SAMPLES])

df: pd.DataFrame | None = None
dataset_id: str | None = None
if uploaded is not None:
    df = read_csv(uploaded)
    dataset_id = f"upload:{uploaded.name}:{uploaded.size}"
elif sample_choice != "(none)":
    df = pd.read_csv(SAMPLES_DIR / SAMPLES[sample_choice])
    dataset_id = f"sample:{sample_choice}"


def prepare_y(series: pd.Series, kind: str) -> pd.Series:
    """Coerce a column to plottable values for its effective kind.

    Bad values become NaN/NaT and simply don't draw — their absence at
    their row position is itself visible in the plot.
    """
    if kind == "numeric":
        return pd.to_numeric(series, errors="coerce")
    if kind == "datetime":
        return as_datetime(series)
    if kind == "text":
        return series.astype(str).str.len().where(series.notna())
    return series  # categorical is handled separately (needs its lanes)


def _axis_ref(axis: str, row: int) -> str:
    """Subplot axis ids for shapes: x/y for row 1, x2/y2 for row 2, ..."""
    return axis if row == 1 else f"{axis}{row}"


def add_categorical_panel(fig: go.Figure, row: int, series: pd.Series, x_all: np.ndarray) -> None:
    """Draw a categorical variable as touching rectangles — a train.

    One rectangle per *run* of consecutive rows sharing a category, in
    that category's lane, each car stretched halfway to its neighbours
    so consecutive cars touch with no gap. A long block of one category
    reads as one solid segment; a one-row typo is a single narrow car
    alone in its lane. Run-length encoding also keeps the shape count
    tiny compared to one rectangle per row.
    """
    categories = list(pd.unique(series.dropna()))
    codes = {cat: i for i, cat in enumerate(categories)}
    frame = pd.DataFrame({"x": x_all, "cat": series}).dropna()
    if len(frame) > MAX_PLOT_POINTS:
        frame = frame.iloc[:: len(frame) // MAX_PLOT_POINTS + 1]
    xs = frame["x"].to_numpy()
    cats = frame["cat"].to_numpy()
    xref, yref = _axis_ref("x", row), _axis_ref("y", row)

    run_start = 0
    for i in range(1, len(cats) + 1):
        if i < len(cats) and cats[i] == cats[run_start]:
            continue  # the run is still going
        code = codes[cats[run_start]]
        if run_start > 0:
            left = xs[run_start] - (xs[run_start] - xs[run_start - 1]) / 2
        else:
            left = xs[run_start] - 0.5
        if i < len(cats):
            right = xs[i - 1] + (xs[i] - xs[i - 1]) / 2
        else:
            right = xs[i - 1] + 0.5
        fig.add_shape(
            type="rect",
            x0=left,
            x1=right,
            y0=code - 0.42,
            y1=code + 0.42,
            fillcolor=CATEGORY_COLORS[code % len(CATEGORY_COLORS)],
            line_width=0,
            xref=xref,
            yref=yref,
        )
        run_start = i

    # Shapes can't hover, so invisible points carry the hover text.
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=[codes[c] for c in cats],
            mode="markers",
            marker={"size": 8, "opacity": 0},
            text=[str(c) for c in cats],
            hovertemplate="row %{x}<br>%{text}<extra></extra>",
        ),
        row=row,
        col=1,
    )
    fig.update_yaxes(
        tickvals=list(codes.values()),
        ticktext=[str(c) for c in categories],
        row=row,
        col=1,
    )


def add_panel(fig: go.Figure, row: int, series: pd.Series, kind: str, x_all: np.ndarray) -> None:
    """Draw one variable into its subplot panel, styled by kind."""
    if kind == "categorical":
        add_categorical_panel(fig, row, series, x_all)
        return

    y = prepare_y(series, kind)
    frame = pd.DataFrame({"x": x_all, "y": y}).dropna()
    if kind == "text":
        frame["label"] = series.astype(str).str.slice(0, 80)
    if len(frame) > MAX_PLOT_POINTS:
        frame = frame.iloc[:: len(frame) // MAX_PLOT_POINTS + 1]
    # Color echoes the value on a viridis scale: continuous variables
    # get a gradient, so a block of wrong-unit rows shows up as a block
    # of wrong color as well as wrong height. Datetimes color by their
    # underlying timestamp number.
    color_values = frame["y"].astype("int64") if kind == "datetime" else frame["y"]
    fig.add_trace(
        go.Scatter(
            x=frame["x"],
            y=frame["y"],
            mode="markers",
            marker={"size": 6, "opacity": 0.75, "color": color_values, "colorscale": "Viridis"},
            text=frame["label"] if kind == "text" else None,
            hovertemplate="row %{x}<br>%{text}<extra></extra>"
            if kind == "text"
            else "row %{x}<br>%{y}<extra></extra>",
        ),
        row=row,
        col=1,
    )
    if kind == "text":
        fig.update_yaxes(title_text="length", row=row, col=1)


def build_figure(df: pd.DataFrame, selected: list[tuple[str, str]]) -> go.Figure:
    """Stack the selected variables in panels over a shared index axis."""
    x_all = np.arange(len(df))
    fig = make_subplots(
        rows=len(selected),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[name for name, _ in selected],
        vertical_spacing=min(0.35 / len(selected), 0.08),
    )
    for i, (name, kind) in enumerate(selected, start=1):
        add_panel(fig, i, df[name], kind, x_all)
    fig.update_xaxes(title_text="index", row=len(selected), col=1)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(
        height=max(180 * len(selected) + 80, 320),
        showlegend=False,
        margin={"l": 50, "r": 10, "t": 40, "b": 40},
    )
    return fig


if df is not None and dataset_id is not None:
    # Per-dataset UI state: effective kinds (user-overridable) and which
    # variables are checked for plotting. Reset when the dataset changes.
    if st.session_state.get("dataset_id") != dataset_id:
        st.session_state["dataset_id"] = dataset_id
        st.session_state["kinds"] = {s.name: s.kind for s in analyze(df)}
        st.session_state["checks"] = {str(col): True for col in df.columns}
        st.session_state["editor_rev"] = 0

    kinds: dict[str, str] = st.session_state["kinds"]
    checks: dict[str, bool] = st.session_state["checks"]

    st.caption(f"{len(df):,} rows × {len(df.columns)} columns")
    summary_rows = [
        {
            "Visualize": checks.get(s.name, True),
            "Variable": s.name,
            "Type": s.kind,
            "Missing": f"{s.n_missing} ({s.pct_missing}%)",
            "Unique": s.n_unique,
            "Summary": format_details(s),
        }
        for s in analyze(df, kinds)
    ]
    edited = st.data_editor(
        pd.DataFrame(summary_rows),
        key=f"vars_{dataset_id}_{st.session_state['editor_rev']}",
        hide_index=True,
        width="stretch",
        disabled=["Variable", "Missing", "Unique", "Summary"],
        column_config={
            "Visualize": st.column_config.CheckboxColumn("Visualize", width=85),
            "Variable": st.column_config.TextColumn("Variable", width=180),
            "Type": st.column_config.SelectboxColumn(
                "Type", options=list(KINDS), required=True, width=130
            ),
            "Missing": st.column_config.TextColumn("Missing", width=105),
            "Unique": st.column_config.NumberColumn("Unique", width=75),
            # Summary gets no width: it takes whatever space is left.
            "Summary": st.column_config.TextColumn("Summary"),
        },
    )

    new_kinds = dict(zip(edited["Variable"], edited["Type"]))
    new_checks = dict(zip(edited["Variable"], edited["Visualize"]))
    st.session_state["checks"] = new_checks
    if new_kinds != kinds:
        # A type override changes the Summary details, so rebuild the
        # table with fresh details (the editor key bump forces the new
        # frame to render; checks and kinds are carried over above).
        st.session_state["kinds"] = new_kinds
        st.session_state["editor_rev"] += 1
        st.rerun()

    selected = [(name, new_kinds[name]) for name in (str(c) for c in df.columns) if new_checks.get(name)]
    if selected:
        st.plotly_chart(
            build_figure(df, selected),
            config={"displaylogo": False, "responsive": True},
        )
    else:
        st.caption("No variables selected.")

    with st.expander("Browse the data"):
        st.dataframe(df, width="stretch")
