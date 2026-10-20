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

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from craneview_core import KINDS, analyze, as_datetime, as_numeric, format_details, truncate_label

# Qualitative palette for categories: Okabe-Ito, designed so that
# neighboring categories stay distinguishable (and colorblind-safe).
# Sequential palettes (viridis, plasma) are for continuous values —
# wrong for categories, whose shades blur into each other.
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]

# Each categorical *variable* gets its own palette family, so a figure
# full of categorical panels isn't the same orange/blue/green over and
# over: variable 1 uses Okabe-Ito, variable 2 Safe, and so on, cycling
# if there are more categorical variables than families. Within a
# variable, categories take the family's colors in order. The
# magenta/pink worked into most families (5th slot) is deliberate —
# pink-forward was requested; small variables keep their original
# first colors, and any variable with 5+ categories picks up a pink.
def _with_pink(family: list[str], pink: str) -> list[str]:
    """Insert a pink into a palette family at the 5th slot."""
    return family[:4] + [pink] + family[4:]


PALETTE_FAMILIES = [
    _with_pink(OKABE_ITO, "#E7298A"),
    _with_pink(list(plotly.colors.qualitative.Safe), "#F781BF"),
    _with_pink(list(plotly.colors.qualitative.Set2), "#E7298A"),
    _with_pink(list(plotly.colors.qualitative.T10), "#F781BF"),
    plotly.colors.qualitative.Bold,
]

# Above this many rows, plots draw a stride sample so the browser stays
# responsive. Summaries always use the full data; only the dots thin out.
MAX_PLOT_POINTS = 20_000

# Datasets longer than this open zoomed to the first INITIAL_WINDOW
# rows, with a range slider + panning to travel the rest — 20,000 rows
# squeezed into one screen width is not QA, it's wallpaper.
INITIAL_WINDOW = 500

# Pixel height budgeted per panel (the figure is PANEL_PX per variable
# plus fixed margins). The categorical label stride uses it to decide
# how many lane labels fit in a panel without colliding.
PANEL_PX = 92

# Categorical bands are 3 lane-heights tall (half-height 1.5 around
# the lane center): adjacent bands overlap by 2/3 of their height,
# like tiles offset by a third — if lane A's band spans 0–12 in some
# unit, B's spans 4–16 and C's 8–20. The fill is translucent so
# overlapped bands blend instead of the top one hiding the rest.
CAT_HALF_HEIGHT = 1.5
CAT_FILL_ALPHA = 0.8

SAMPLES = {
    "Messy plants (100 rows, planted errors)": "messy_plants.csv",
    "Iris (150 rows)": "iris.csv",
    "Cars — mpg (398 rows)": "mpg.csv",
    "Field trial (2,000 rows)": "field_trial_2000.csv",
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
    if kind in ("numeric", "log numeric"):
        return as_numeric(series)
    if kind == "datetime":
        return as_datetime(series)
    if kind == "text":
        return series.astype(str).str.len().where(series.notna())
    return series  # categorical is handled separately (needs its lanes)


def _axis_ref(axis: str, row: int) -> str:
    """Subplot axis ids for shapes: x/y for row 1, x2/y2 for row 2, ..."""
    return axis if row == 1 else f"{axis}{row}"


def _rgba(color: str, alpha: float) -> str:
    """Add an alpha channel to a Plotly color.

    Palette colors arrive in two formats — '#E69F00' (hand-written and
    some Plotly palettes) and 'rgb(230,159,0)' (other Plotly palettes,
    e.g. qualitative Safe) — so parse whichever shows up.
    """
    if color.startswith("#"):
        r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    else:  # 'rgb(r, g, b)'
        r, g, b = (int(part) for part in color[color.index("(") + 1 : color.index(")")].split(","))
    return f"rgba({r},{g},{b},{alpha})"


def add_categorical_panel(
    fig: go.Figure, row: int, series: pd.Series, x_all: np.ndarray, palette: list[str]
) -> None:
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

    # Collect the runs first; they are painted lane by lane below.
    runs: list[tuple[int, float, float]] = []  # (lane code, left, right)
    run_start = 0
    for i in range(1, len(cats) + 1):
        if i < len(cats) and cats[i] == cats[run_start]:
            continue  # the run is still going
        if run_start > 0:
            left = xs[run_start] - (xs[run_start] - xs[run_start - 1]) / 2
        else:
            left = xs[run_start] - 0.5
        if i < len(cats):
            right = xs[i - 1] + (xs[i] - xs[i - 1]) / 2
        else:
            right = xs[i - 1] + 0.5
        runs.append((codes[cats[run_start]], float(left), float(right)))
        run_start = i

    # Paint order keeps the layering consistent instead of depending
    # on which run happens to come first in the data. NOTE: shapes are
    # collected and assigned in ONE batch below — calling add_shape()
    # per run is quadratic in plotly (each call revalidates the whole
    # layout), which hangs on a few thousand runs; one assignment of
    # the full tuple is near-instant.
    new_shapes = [
        go.layout.Shape(
            type="rect",
            x0=left,
            x1=right,
            y0=code - CAT_HALF_HEIGHT,
            y1=code + CAT_HALF_HEIGHT,
            fillcolor=_rgba(palette[code % len(palette)], CAT_FILL_ALPHA),
            line_width=0,
            xref=xref,
            yref=yref,
        )
        for code, left, right in sorted(runs)
    ]
    fig.layout.shapes = tuple(fig.layout.shapes) + tuple(new_shapes)

    # Shapes can't hover, so invisible points carry the hover text.
    fig.add_trace(
        go.Scattergl(
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
    # With many categories the lane labels collide vertically (a panel
    # fits roughly one label per 13px), so label every Nth lane and
    # leave the rest to hover, which always shows the full value.
    label_stride = max(1, math.ceil(len(categories) * 13 / PANEL_PX))
    shown = range(0, len(categories), label_stride)
    fig.update_yaxes(
        tickvals=[codes[categories[i]] for i in shown],
        # Axis labels are truncated (the full value is on hover): one
        # pasted-paragraph category would otherwise shove the whole
        # plot to the right.
        ticktext=[truncate_label(str(categories[i])) for i in shown],
        # Pad the range slightly so the first/last bands aren't clipped.
        range=[-CAT_HALF_HEIGHT - 0.05, len(categories) - 1 + CAT_HALF_HEIGHT + 0.05],
        # No horizontal line through the first lane (the y=0 zeroline).
        zeroline=False,
        row=row,
        col=1,
    )
    # No x-axis line on categorical panels — the bands are the picture.
    fig.update_xaxes(showline=False, zeroline=False, ticks="", row=row, col=1)


def add_panel(
    fig: go.Figure,
    row: int,
    series: pd.Series,
    kind: str,
    x_all: np.ndarray,
    palette: list[str] | None = None,
) -> None:
    """Draw one variable into its subplot panel, styled by kind."""
    if kind == "categorical":
        add_categorical_panel(fig, row, series, x_all, palette or OKABE_ITO)
        return

    y = prepare_y(series, kind)
    frame = pd.DataFrame({"x": x_all, "y": y}).dropna()
    if kind == "text":
        frame["label"] = series.astype(str).str.slice(0, 80)
    if len(frame) > MAX_PLOT_POINTS:
        frame = frame.iloc[:: len(frame) // MAX_PLOT_POINTS + 1]
    # Color echoes the value on the plasma scale: continuous variables
    # get a gradient, so a block of wrong-unit rows shows up as a block
    # of wrong color as well as wrong height. Datetimes color by their
    # underlying timestamp number.
    color_values = frame["y"].astype("int64") if kind == "datetime" else frame["y"]
    fig.add_trace(
        go.Scattergl(
            x=frame["x"],
            y=frame["y"],
            mode="markers",
            marker={"size": 6, "opacity": 0.75, "color": color_values, "colorscale": "Plasma"},
            text=frame["label"] if kind == "text" else None,
            hovertemplate="row %{x}<br>%{text}<extra></extra>"
            if kind == "text"
            else "row %{x}<br>%{y}<extra></extra>",
        ),
        row=row,
        col=1,
    )
    if kind == "log numeric":
        # Log-scale y axis. Zero/negative values can't exist on a log
        # axis, so they don't draw — the summary table still counts
        # them (min/mean/median are computed on the raw values).
        fig.update_yaxes(type="log", row=row, col=1)


def build_figure(df: pd.DataFrame, selected: list[tuple[str, str]]) -> go.Figure:
    """Stack the selected variables in panels over a shared index axis."""
    x_all = np.arange(len(df))
    fig = make_subplots(
        rows=len(selected),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.18 / len(selected), 0.022),
    )
    cat_number = 0  # which categorical variable this is (for palettes)
    for i, (name, kind) in enumerate(selected, start=1):
        palette = None
        if kind == "categorical":
            palette = PALETTE_FAMILIES[cat_number % len(PALETTE_FAMILIES)]
            cat_number += 1
        add_panel(fig, i, df[name], kind, x_all, palette)
        # Variable names label the left side (the y-axis title), not the
        # top of each panel. Text plots get a hint that y is length.
        label = f"{name} (length)" if kind == "text" else name
        fig.update_yaxes(title_text=label, row=i, col=1)
    fig.update_xaxes(title_text="index", row=len(selected), col=1)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    if len(df) > INITIAL_WINDOW:
        # Open on the first window of rows; the range slider under the
        # bottom axis (plus dragging to pan, scroll to zoom) travels
        # the rest. The axes are shared, so every panel moves together.
        for i in range(1, len(selected) + 1):
            fig.update_xaxes(range=[-0.5, INITIAL_WINDOW - 0.5], row=i, col=1)
        fig.update_xaxes(
            rangeslider={
                "visible": True,
                # Slim and bright — the default slider is tall and
                # murky against the dark theme — and its travel is
                # clamped to the data, so the window can't be dragged
                # into negative index numbers.
                "thickness": 0.07,
                "bgcolor": "rgba(148,163,184,0.30)",
                "bordercolor": "rgba(226,232,240,0.9)",
                "borderwidth": 1,
                "range": [0, len(df) - 1],
            },
            # Index numbers for reference, just above the slider.
            showticklabels=True,
            tickfont={"size": 13},
            row=len(selected),
            col=1,
        )
    fig.update_layout(
        height=max(PANEL_PX * len(selected) + 60, 280),
        showlegend=False,
        dragmode="pan",
        # Clicking a point selects it (without this, clicks only fire
        # hover events and Streamlit never hears about them).
        clickmode="event+select",
        margin={"l": 70, "r": 10, "t": 20, "b": 40},
    )
    return fig


def _row_from_point(fig: go.Figure, point: dict) -> int | None:
    """Row index for a clicked chart point.

    Every trace's x is the row index, so the point's own x is the
    answer; if a payload ever lacks x, look the point up in the trace
    it came from instead.
    """
    x = point.get("x")
    if x is not None:
        return int(x)
    curve = point.get("curve_number", point.get("curveNumber"))
    index = point.get("point_index", point.get("pointIndex"))
    if curve is not None and index is not None and curve < len(fig.data):
        xs = fig.data[curve].x
        if xs is not None and index < len(xs):
            return int(xs[index])
    return None


def _step_focus(delta: int, n_rows: int) -> None:
    """Move the focused row by delta, clamped to the data."""
    current = st.session_state.get("focus_row")
    if current is not None:
        st.session_state["focus_row"] = min(max(current + delta, 0), n_rows - 1)


def _clear_focus() -> None:
    st.session_state["focus_row"] = None


if df is not None and dataset_id is not None:
    # Per-dataset UI state: effective kinds (user-overridable) and which
    # variables are checked for plotting. Reset when the dataset changes.
    if st.session_state.get("dataset_id") != dataset_id:
        st.session_state["dataset_id"] = dataset_id
        st.session_state["kinds"] = {s.name: s.kind for s in analyze(df)}
        st.session_state["checks"] = {str(col): True for col in df.columns}
        st.session_state["editor_rev"] = 0
        st.session_state["focus_row"] = None
        st.session_state["chart_signature"] = None

    kinds: dict[str, str] = st.session_state["kinds"]
    checks: dict[str, bool] = st.session_state["checks"]
    # A saved kind can outlive the app version that offered it (types
    # get merged/renamed between releases); anything no longer on the
    # menu falls back to the detected kind instead of breaking the
    # editor's dropdown.
    detected_kinds = {s.name: s.kind for s in analyze(df)}
    kinds = {name: (k if k in KINDS else detected_kinds[name]) for name, k in kinds.items()}

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
                # The ▾ is in the label because Streamlit's editor only
                # shows the dropdown arrow once a cell is clicked — at
                # rest, nothing marks the column as a dropdown.
                "Type ▾",
                options=list(KINDS),
                required=True,
                width=145,
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
        fig = build_figure(df, selected)
        event = st.plotly_chart(
            fig,
            key="craneview_plot",
            on_select="rerun",
            selection_mode="points",
            config={"displaylogo": False, "responsive": True, "scrollZoom": True},
        )
        points = event.selection.points if event is not None and event.selection else []
        if points:
            # Only a *changed* selection moves the focus — otherwise
            # the still-reported last click would keep snapping the
            # focus back after the user steps to a neighbouring row.
            signature = tuple(
                (
                    p.get("curve_number", p.get("curveNumber")),
                    p.get("point_index", p.get("pointIndex")),
                    p.get("x"),
                )
                for p in points
            )
            if signature != st.session_state.get("chart_signature"):
                st.session_state["chart_signature"] = signature
                row = _row_from_point(fig, points[0])
                if row is not None:
                    st.session_state["focus_row"] = row
    else:
        st.caption("No variables selected.")

    # The focused row: the full record behind a clicked point. (The
    # data table itself can't be scrolled to a row — Streamlit offers
    # no API for that — so the record comes to a strip of its own,
    # with stepping to walk the neighbourhood of a suspicious point.)
    focus = st.session_state.get("focus_row")
    if focus is None or not 0 <= focus < len(df):
        st.caption("Click any point in the plot to inspect its full row here.")
    else:
        nav = st.columns([1, 4, 1, 1])
        nav[0].button(
            "← Prev", key="prev_row", disabled=focus == 0,
            on_click=_step_focus, args=(-1, len(df)),
        )
        nav[1].markdown(f"**Row {focus}** — full record ({len(df):,} rows total)")
        nav[2].button(
            "Next →", key="next_row", disabled=focus == len(df) - 1,
            on_click=_step_focus, args=(1, len(df)),
        )
        nav[3].button("Clear", key="clear_row", on_click=_clear_focus)
        record = df.iloc[focus].rename("Value")
        record.index.name = "Variable"
        st.dataframe(record.to_frame(), width="stretch", height=min(38 + 35 * len(record), 420))

    with st.expander("Browse the data"):
        st.dataframe(df, width="stretch")
