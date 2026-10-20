"""CraneView core logic: column type detection and per-column summaries.

This module has no web/UI code in it on purpose. The Streamlit app
(app.py) is a thin shell around these functions, so the QA logic can be
read, tested, and reused on its own.

Design idea: a CSV uploaded for QA gets plotted in row-index order
(see app.py). Plotting in row order is the whole trick — data-entry
anomalies like a stray 999, a block of values entered in the wrong
unit, or two swapped rows are invisible in a histogram but jump out
of a run chart. These functions decide *what kind* of plot each
column gets and compute the combined summary table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

# A numeric column with at most this many distinct values is almost
# certainly a code or a rating scale (1-5 stars, treatment group 0/1/2),
# so it is plotted and summarized as categorical instead of numeric.
MAX_CATEGORICAL_CODES = 12

# An object column counts as categorical if it has few distinct values
# outright, or if its values repeat a lot (low cardinality ratio).
MAX_CATEGORICAL_UNIQUE = 20
MAX_CATEGORICAL_RATIO = 0.05

# A string column is treated as datetime only if nearly all of its
# non-missing values parse as dates. The threshold is strict because a
# loose parse turns ID numbers and codes into nonsense dates.
DATETIME_PARSE_THRESHOLD = 0.8


@dataclass
class ColumnSummary:
    """Everything the UI needs to show for one column."""

    name: str
    kind: str  # one of KINDS (craneview_core.KINDS)
    n: int
    n_missing: int
    pct_missing: float
    n_unique: int
    details: dict = field(default_factory=dict)


def detect_kind(series: pd.Series) -> str:
    """Classify one column as numeric, categorical, datetime, or text.

    Order matters: real dtypes win first, then a strict datetime parse
    for string columns, then cardinality decides categorical vs text.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_bool_dtype(series):
        return "categorical"

    if pd.api.types.is_numeric_dtype(series):
        if series.nunique(dropna=True) <= MAX_CATEGORICAL_CODES:
            return "categorical"
        return "numeric"

    # From here the column is strings/objects (or empty).
    non_null = series.dropna()
    if non_null.empty:
        return "text"

    if _mostly_dates(non_null):
        return "datetime"

    n_unique = non_null.nunique()
    if n_unique <= MAX_CATEGORICAL_UNIQUE or n_unique / len(non_null) <= MAX_CATEGORICAL_RATIO:
        return "categorical"
    return "text"


def _mostly_dates(non_null: pd.Series) -> bool:
    """True if >= DATETIME_PARSE_THRESHOLD of the values parse as dates."""
    # Only attempt the parse on columns that look date-ish; this keeps
    # plain numbers-as-strings ("1042", "8871") from becoming datetimes.
    sample = non_null.astype(str).head(200)
    if not sample.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", regex=True).any():
        return False
    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    return parsed.notna().mean() >= DATETIME_PARSE_THRESHOLD


def as_datetime(series: pd.Series) -> pd.Series:
    """Coerce a column to datetime for plotting (bad values become NaT)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce", format="mixed")


KINDS = ("numeric", "log numeric", "stripped numeric", "categorical", "datetime", "text")

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _first_number(value: object) -> float:
    """The first number inside a text value ('PLT-1042' -> 1042.0).

    A '-' glued to a letter is a separator, not a minus sign, so
    'PLT-1042' is 1042 while '-3.5 C' and 'temp -3.5' stay negative.
    Thousands separators are ignored ('1,042' -> 1042).
    """
    text = str(value).replace(",", "")
    match = _NUMBER_RE.search(text)
    if match is None:
        return float("nan")
    found = match.group()
    if found.startswith("-") and match.start() > 0 and text[match.start() - 1].isalpha():
        found = found[1:]
    return float(found)


def extract_numbers(series: pd.Series) -> pd.Series:
    """Pull the first number out of each value of a text column.

    Backs the 'stripped numeric' type: a column like 'PLT-1042',
    '12.5 kg', 'P07' becomes the numbers inside it. Values with no
    number become NaN — they don't plot, and the gap is visible, the
    same contract as the other coercions.
    """
    return series.map(_first_number)


def summarize(series: pd.Series, kind: str | None = None) -> ColumnSummary:
    """Compute one column's summary stats.

    `kind` is the *effective* type: the detected type, unless the user
    overrode it in the UI. Counts (missing, unique) always describe the
    raw column; the detail stats follow the effective kind, coercing
    values as needed (a text column forced to numeric gets its stats
    from whatever actually parses as numbers).
    """
    if kind is None:
        kind = detect_kind(series)
    n = len(series)
    n_missing = int(series.isna().sum())
    summary = ColumnSummary(
        name=str(series.name),
        kind=kind,
        n=n,
        n_missing=n_missing,
        pct_missing=round(100 * n_missing / n, 1) if n else 0.0,
        n_unique=int(series.nunique(dropna=True)),
    )

    non_null = series.dropna()
    if kind in ("numeric", "log numeric", "stripped numeric"):
        raw = (
            extract_numbers(non_null)
            if kind == "stripped numeric"
            else pd.to_numeric(non_null, errors="coerce")
        )
        values = raw.dropna()
        summary.details = {
            "min": _clean_number(values.min()),
            "max": _clean_number(values.max()),
            "mean": _clean_number(values.mean()),
            "median": _clean_number(values.median()),
        }
    elif kind == "categorical":
        top = non_null.value_counts().head(5)
        summary.details = {"top_values": {str(k): int(v) for k, v in top.items()}}
    elif kind == "datetime":
        dates = as_datetime(non_null)
        summary.details = {
            "earliest": str(dates.min().date()) if dates.notna().any() else None,
            "latest": str(dates.max().date()) if dates.notna().any() else None,
        }
    else:  # text
        lengths = non_null.astype(str).str.len()
        summary.details = {
            "min_length": int(lengths.min()) if len(lengths) else 0,
            "max_length": int(lengths.max()) if len(lengths) else 0,
            "examples": [str(v) for v in non_null.head(3)],
        }
    return summary


def _clean_number(value) -> float | None:
    """Round floats for display; pass NaN through as None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return round(float(value), 4)


def analyze(df: pd.DataFrame, kinds: dict[str, str] | None = None) -> list[ColumnSummary]:
    """Summarize every column, in the order the columns appear in the CSV.

    `kinds` optionally maps column name -> effective kind (user
    overrides from the UI); columns not in the map use detected kinds.
    """
    kinds = kinds or {}
    return [summarize(df[col], kind=kinds.get(str(col))) for col in df.columns]


def format_details(summary: ColumnSummary) -> str:
    """Render a summary's detail stats as one compact line for the table.

    Examples: "min 20.2 · max 9,999 · mean 251.7 · median 108.8" for
    numeric, "F 84 · M 34 · f 2" for categorical, a date range for
    datetime, a length range for text.
    """
    d = summary.details
    if summary.kind in ("numeric", "log numeric", "stripped numeric"):
        return " · ".join(
            f"{label} {_format_number(d.get(key))}"
            for label, key in (("min", "min"), ("max", "max"), ("mean", "mean"), ("median", "median"))
        )
    if summary.kind == "categorical":
        top = d.get("top_values", {})
        line = " · ".join(f"{truncate_label(value)} {count}" for value, count in top.items())
        extra = summary.n_unique - len(top)
        return f"{line} · +{extra} more" if extra > 0 else line
    if summary.kind == "datetime":
        return f"{d.get('earliest')} → {d.get('latest')}"
    return f"length {d.get('min_length')}–{d.get('max_length')} chars"


def _format_number(value: float | None) -> str:
    """Compact number for display: thousands separators, <=4 sig figs."""
    if value is None:
        return "—"
    return f"{value:,.4g}"


def truncate_label(text: str, limit: int = 28) -> str:
    """Cap a category label so one pasted-paragraph value can't blow
    out a table cell or shove a plot sideways as an axis tick label.
    The full value still shows on hover."""
    return text if len(text) <= limit else text[: limit - 1] + "…"
