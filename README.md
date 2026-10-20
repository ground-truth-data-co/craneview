# CraneView

Eyeball a CSV for data-entry anomalies. Upload a file and CraneView plots
**every column in row order**, one at a time, in the order the columns
appear in the file.

Row order is the point. A histogram tells you a column has weird values;
a run chart (value vs. row number) tells you *where* they are — the stray
999 on row 412, the block of rows 200–260 entered in centimeters instead
of meters, the two rows whose values got swapped. Those patterns are how
data-entry errors actually look, and they're what this tool is for.

## What it does

- **One summary table for all variables**: name, missing count/percent,
  unique values, and a compact summary line (range/mean for numeric, top
  values for categorical, date range for datetime, length range for text).
- **The data type is a dropdown in that table.** Detection is a heuristic
  (numeric / categorical / datetime / text); override any column whose
  type it got wrong and the summary and plot update to match. Two extra
  **numeric** sees through text wrappers: set a column like
  `PLT-1042` or `12.5 kg` to numeric and it plots the numbers inside
  (values containing no number are rejected as missing). **log
  numeric** is the same numbers on a log-scale axis.
- **A checkbox per variable** picks which ones are plotted.
- **One stacked plot**: selected variables get one panel each, stacked
  over a shared row-index axis (0, 1, 2, 3, …). Numeric and datetime
  panels plot the value against row number, colored by value on the
  plasma scale; categorical panels draw one lane per category filled
  with touching rectangles — one per run of consecutive rows — in a
  qualitative palette (a different palette family per variable; a
  typo'd level like `f` vs `F` gets its own lane, color, and visibly
  tiny train car); text panels plot string length, since the text
  itself can't be a y-axis.
- **Bundled samples** if you have no file handy: a 100-row plant dataset
  with planted entry errors, iris, the classic mpg cars data, and a
  2,000-row field trial.

Your file is processed in memory for the session only. Nothing is stored.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Project layout

- `app.py` — the Streamlit app: upload, browse table, per-column sections.
  Deliberately thin.
- `craneview_core.py` — type detection and summaries, no UI code. This is
  where the QA opinions live; it's importable and testable on its own.

## Deploy

It's a standard Streamlit app: push this folder to a GitHub repo and point
[Streamlit Community Cloud](https://streamlit.io/cloud) (or a Hugging Face
Space with the Streamlit SDK) at `app.py`.
