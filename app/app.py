# app.py
# Streamlit application to parse and visualize survey crosstab data
# Supports:
#   • Numeric crosstabs (counts + percent)
#   • Open-ended questions named with `_text` anywhere in the label:
#       - whole-sheet names (e.g., "S2Q12_text", "S2Q12_text – comments")
#       - column/row headers where any cell contains a token like "S2Q12_text:"
#   • LLM summaries via local Ollama with privacy guardrails
#   • Proper 100% normalization by response
#   • Excludes "Total" from charts to avoid axis distortion

import io
import re
import requests
from collections import Counter
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------
# Constants / markers
# -------------------------
START_MARKERS = [
    "total count [answering]",
    "total count (answering)",
    "total count answering",
]
STOP_MARKERS = [
    "total count [all]",
    "total count (all)",
    "total count all",
    "overall stat test of percentages",
    "overall stat test",
    "overall stats",
]
SQ_TOKEN = re.compile(r"\b[SA]\d+Q\d+\b", re.IGNORECASE)

STOPWORDS = set("""
a an and are as at be but by for from has have if in into is it its no not of on or so such that the their then there these they this to was we were what when where which who will with you your
""".split())

# -------------------------
# Helpers (general)
# -------------------------
def _clean_str(x) -> str:
    return "" if (x is None or (isinstance(x, float) and pd.isna(x))) else str(x).strip()

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).replace("\u00a0", " ")).strip()

_TEXT_TOKEN_RE = re.compile(r"\b([A-Za-z0-9]+_text)\b", re.IGNORECASE)

def _extract_text_tag(s: str) -> Optional[str]:
    """Return first token like 'S2Q12_text' from a string (case-insensitive)."""
    m = _TEXT_TOKEN_RE.search(_norm_ws(s))
    return m.group(1).strip() if m else None

def _is_text_question(name: str) -> bool:
    """Open-ended if it contains a token ending with '_text'."""
    return _extract_text_tag(name) is not None

# --- privacy scrubbing before LLM ---
EMAIL_RE   = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE_RE   = re.compile(r'\b(?:\+?\d{1,3}[\s\-\.]?)?(?:\(?\d{3}\)?[\s\-\.]?)?\d{3}[\s\-\.]?\d{4}\b')
HANDLE_RE  = re.compile(r'@\w+')
URL_RE     = re.compile(r'https?://\S+')
NUMID_RE   = re.compile(r'\b\d{4,}\b')

def sanitize_for_privacy(text: str) -> str:
    t = EMAIL_RE.sub("[email_removed]", text)
    t = PHONE_RE.sub("[phone_removed]", t)
    t = HANDLE_RE.sub("[handle_removed]", t)
    t = URL_RE.sub("[url_removed]", t)
    t = NUMID_RE.sub("[id_removed]", t)
    return t

def sanitize_list(texts: List[str]) -> List[str]:
    return [sanitize_for_privacy(str(t)) for t in texts if str(t).strip()]

# -------------------------
# Ollama LLM summarization
# -------------------------
def ollama_summarize(texts: List[str], model: str = "llama3.1:8b", host: str = "http://localhost:11434",
                     max_tokens: int = 600, temperature: float = 0.2) -> str:
    if not texts:
        return "No responses to summarize."
    joined = "\n- " + "\n- ".join(sanitize_list(texts))
    system = (
        "You are a research assistant. Summarize open-ended survey responses into concise themes. "
        "STRICT PRIVACY: Do not reveal or infer protected attributes or identities (gender, sexuality, "
        "nationality, ethnicity, religion, disability, immigration status), names, departments, or small groups. "
        "Generalize to the core issue. Output 5–10 bullet themes with 1–2 sentence descriptions each, "
        "then 3–6 actionable recommendations. Avoid direct quotes or specifics that could identify someone."
    )
    user = f"Responses:{joined}\n\nReturn:\n- Themes (bulleted)\n- Actionable steps"
    payload = {
        "model": model,
        "prompt": f"{system}\n\n{user}",
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "stream": False,
    }
    try:
        r = requests.post(f"{host}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        return (r.json().get("response") or "").strip() or "LLM returned no content."
    except Exception as e:
        return f"⚠️ Could not reach Ollama ({e})."

# -------------------------
# Crosstab parsing helpers
# -------------------------
def _find_all_start_rows(df_raw: pd.DataFrame) -> List[int]:
    starts: List[int] = []
    nrows, ncols = df_raw.shape
    if ncols == 0:
        return starts
    for i in range(nrows):
        row = df_raw.iloc[i].tolist()
        if any(m in _clean_str(v).lower() for v in row for m in START_MARKERS):
            if not starts or i != starts[-1]:
                starts.append(i)
    deduped = []
    for i in starts:
        if not deduped or i - deduped[-1] > 1:
            deduped.append(i)
    return deduped

def _extract_left_question(df_raw: pd.DataFrame, start_idx: int, max_up: int = 200) -> Optional[str]:
    best_any = None
    for i in range(start_idx, max(-1, start_idx - max_up), -1):
        row_vals = df_raw.iloc[i].tolist()
        boundary = any((m in _clean_str(v).lower()) for v in row_vals for m in (START_MARKERS + STOP_MARKERS))
        if boundary and i != start_idx:
            break
        s = _clean_str(df_raw.iat[i, 0] if df_raw.shape[1] > 0 else "")
        if not s or "total count" in s.lower():
            continue
        if best_any is None:
            best_any = s
        if SQ_TOKEN.search(s):
            return s
    if best_any:
        return best_any
    for i in range(start_idx, max(-1, start_idx - max_up), -1):
        s = _clean_str(df_raw.iat[i, 0] if df_raw.shape[1] > 0 else "")
        if len(s) >= 25 and any(ch.isalpha() for ch in s) and "total count" not in s.lower():
            return s
    return None

def _find_header_row_above(df_raw: pd.DataFrame, start_idx: int, search_up: int = 200) -> Optional[str]:
    nrows = len(df_raw)
    candidates = []
    for k in range(1, min(search_up, start_idx) + 1):
        i = start_idx - k
        row = df_raw.iloc[i]
        nonnulls = row.notna().sum()
        joined = " | ".join([_clean_str(x) for x in row.tolist()])
        has_total = (" total " in f" {joined.lower()} ") or joined.lower().startswith("total") or joined.lower().endswith(" total")
        candidates.append((i, has_total, nonnulls, joined))
        if any((m in _clean_str(v).lower()) for v in row.tolist() for m in START_MARKERS):
            break
    with_total = [c for c in candidates if c[1]]
    if with_total:
        return max(with_total, key=lambda x: (x[0], x[2]))[3]
    if candidates:
        return max(candidates, key=lambda x: (x[2], x[0]))[3]
    return None

def _find_global_header_row(df_raw: pd.DataFrame) -> Optional[str]:
    nrows, ncols = df_raw.shape
    for i in range(min(120, nrows)):
        row = df_raw.iloc[i].tolist()
        for j, v in enumerate(row):
            if _clean_str(v).lower() == "total":
                right = row[j: j + 16]
                names = [_clean_str(x) for x in right if _clean_str(x) != ""]
                if len(names) >= 3:
                    return " | ".join([_clean_str(x) for x in row])
    return None

def _find_header_panels(header_row: str) -> List[Tuple[int, List[int], List[str]]]:
    header_list = header_row.split(" | ")
    n = len(header_list)
    totals = [j for j, v in enumerate(header_list) if _clean_str(v).lower() == "total"]
    panels: List[Tuple[int, List[int], List[str]]] = []
    for tpos in totals:
        demo_cols, demo_names = [], []
        j = tpos
        while j < n:
            val = _clean_str(header_list[j])
            if val == "":
                break
            demo_cols.append(j)
            demo_names.append(re.sub(r"\s+", " ", val))
            j += 1
        if len(demo_cols) >= 2:
            panels.append((tpos, demo_cols, demo_names))
    return panels

def _choose_response_col(df_raw: pd.DataFrame, start_idx: int, panel_start_col: int) -> int:
    nrows, ncols = df_raw.shape
    left_candidates = [c for c in range(0, panel_start_col)] or [0]
    def _text_count(col):
        cnt = 0
        for r in range(start_idx + 1, min(start_idx + 60, nrows)):
            s = _clean_str(df_raw.iat[r, col] if col < ncols else "")
            if s and not s.replace(".", "", 1).isdigit():
                cnt += 1
        return cnt
    return max(left_candidates, key=_text_count)

# --- Normalize by response for 100% stacks (so each x tick sums to 100) ---
def _normalize_by_response(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    denom = tmp.groupby("response")["percent"].transform(lambda s: s.sum() if pd.notna(s).any() else np.nan)
    tmp["norm_percent"] = np.where((denom > 0) & pd.notna(tmp["percent"]), (tmp["percent"] / denom) * 100.0, np.nan)
    return tmp

def _parse_block_all_panels(df_raw: pd.DataFrame, start_idx: int, global_header: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, Optional[float]], int]:
    nrows, ncols = df_raw.shape
    end_idx = None
    for j in range(start_idx + 1, nrows):
        row = df_raw.iloc[j].tolist()
        if any(m in _clean_str(v).lower() for v in row for m in STOP_MARKERS):
            end_idx = j; break
        if any(m in _clean_str(v).lower() for v in row for m in START_MARKERS):
            end_idx = j; break
    if end_idx is None:
        end_idx = nrows

    question = _extract_left_question(df_raw, start_idx) or "Question"
    header_row_str = _find_header_row_above(df_raw, start_idx, search_up=400)
    panels = _find_header_panels(header_row_str) if header_row_str else []
    if not panels and global_header:
        panels = _find_header_panels(global_header)
    if not panels:
        return pd.DataFrame(columns=["question","response","demographic","count","percent"]), {}, end_idx

    overall_val = None
    for k in range(max(0, start_idx - 6), min(nrows, start_idx + 6)):
        row = df_raw.iloc[k].tolist()
        if any("overall stat" in _clean_str(v).lower() for v in row):
            for v in row:
                vs = _clean_str(v)
                try:
                    overall_val = float(vs); break
                except Exception:
                    pass
            break

    records = []
    for panel_total_pos, demo_cols, demo_names in panels:
        resp_col = _choose_response_col(df_raw, start_idx, panel_total_col=panel_total_pos) if False else _choose_response_col(df_raw, start_idx, panel_total_pos)
        i = start_idx + 2
        while i + 1 < end_idx:
            resp = _clean_str(df_raw.iat[i, resp_col] if resp_col < ncols else "")
            numeric_ok = False
            row_counts, row_perc = [], []
            for col_idx in demo_cols:
                count_str = _clean_str(df_raw.iat[i, col_idx] if col_idx < ncols else "")
                perc_str  = _clean_str(df_raw.iat[i + 1, col_idx] if (i + 1) < nrows and col_idx < ncols else "")
                try:
                    cval = float(count_str); row_counts.append(cval)
                except Exception:
                    row_counts.append(np.nan)
                try:
                    f = float(perc_str); row_perc.append(f); numeric_ok = True
                except Exception:
                    row_perc.append(np.nan)

            if resp and numeric_ok:
                for name, cnt, frac in zip(demo_names, row_counts, row_perc):
                    if pd.notna(frac) or pd.notna(cnt):
                        records.append([question, resp, name, cnt, (frac * 100.0) if pd.notna(frac) else np.nan])
                i += 2
            else:
                i += 1

    tidy = pd.DataFrame(records, columns=["question","response","demographic","count","percent"])
    overall_stats = {question: overall_val}
    return tidy, overall_stats, end_idx

def parse_crosstab_sheet(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[float]]]:
    starts = _find_all_start_rows(df_raw)
    global_header = _find_global_header_row(df_raw)
    all_rows = []
    per_q_overall: Dict[str, Optional[float]] = {}
    for s in starts:
        tidy_b, overall_b, _ = _parse_block_all_panels(df_raw, s, global_header)
        if not tidy_b.empty:
            all_rows.append(tidy_b)
            per_q_overall.update(overall_b)
    tidy_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=["question","response","demographic","count","percent"])
    return tidy_df, per_q_overall

# --- harvesters for open-ended responses ---
def harvest_text_columns(df_raw: pd.DataFrame, scan_depth: int = 8) -> Dict[str, List[str]]:
    """Check the first few non-empty cells of each column for a *_text token; collect below."""
    found: Dict[str, List[str]] = {}
    if df_raw.shape[1] == 0:
        return found
    for c in range(df_raw.shape[1]):
        col = df_raw.iloc[:, c]
        non_empty_idxs = [i for i, v in enumerate(col) if _clean_str(v)]
        if not non_empty_idxs:
            continue
        header_idx = None
        text_tag = None
        for i in non_empty_idxs[:scan_depth]:
            raw = _clean_str(col.iat[i])
            tag = _extract_text_tag(raw)
            if tag:
                header_idx = i
                text_tag = tag
                break
        if header_idx is None:
            continue
        responses = col.iloc[header_idx + 1 :].dropna().astype(str).map(_norm_ws)
        responses = [r for r in responses.tolist() if r]
        if responses:
            key = text_tag if text_tag not in found else f"{text_tag}__col{c}"
            found[key] = responses
    return found

def harvest_text_anywhere(df_raw: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Scan every cell. If a *_text token is found, collect responses:
      • all non-empty cells BELOW in the same column
      • all non-empty cells TO THE RIGHT in the same row
    This captures both column- and row-oriented response layouts.
    """
    found: Dict[str, List[str]] = {}
    nrows, ncols = df_raw.shape
    for i in range(nrows):
        row_vals = df_raw.iloc[i].tolist()
        for j in range(ncols):
            tag = _extract_text_tag(row_vals[j]) if j < len(row_vals) else None
            if not tag:
                continue
            down = df_raw.iloc[i+1:, j].dropna().astype(str).map(_norm_ws)
            right = df_raw.iloc[i, j+1:].dropna().astype(str).map(_norm_ws)
            resp = pd.concat([down, right], ignore_index=True)
            responses = [r for r in resp.tolist() if r]
            if responses:
                key = tag if tag not in found else f"{tag}__r{i}c{j}"
                found[key] = responses
    return found

def parse_workbook(file_like_or_path: Union[str, io.BytesIO]) -> Tuple[pd.DataFrame, Dict[str, Optional[float]], Dict[str, List[str]]]:
    """
    Reads an Excel workbook (.xlsx, all sheets) or a CSV file and returns
    parsed data in a tidy format.
    Collects open-ended questions from:
      • sheets whose names contain a *_text token
      • columns whose early cells contain a *_text token
      • ANY cell in ANY sheet that contains a *_text token (row or column layouts)
    """
    def _df_clean(df):
        return df.applymap(lambda x: np.nan if _clean_str(x) == "" else x)

    fname = getattr(file_like_or_path, "name", None)
    is_csv = isinstance(file_like_or_path, str) and file_like_or_path.lower().endswith(".csv")
    if isinstance(file_like_or_path, io.BytesIO) and fname:
        is_csv = fname.lower().endswith(".csv")

    if is_csv:
        df_raw = pd.read_csv(file_like_or_path, header=None, dtype=str)
        df_raw = _df_clean(df_raw)
        tidy_df, overall_stats = parse_crosstab_sheet(df_raw)
        text_questions = {}
        # columns + anywhere for CSV too
        text_questions.update(harvest_text_columns(df_raw))
        for k, v in harvest_text_anywhere(df_raw).items():
            if k not in text_questions:
                text_questions[k] = v
        return tidy_df, overall_stats, text_questions

    # Excel
    xls = pd.ExcelFile(file_like_or_path, engine="openpyxl")
    all_rows = []
    overall_stats: Dict[str, Optional[float]] = {}
    text_questions: Dict[str, List[str]] = {}

    for sheet in xls.sheet_names:
        df_raw = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=str)
        df_raw = _df_clean(df_raw)

        # 1) whole-sheet open-ended if sheet name contains *_text
        tag_from_sheet = _extract_text_tag(sheet)
        if tag_from_sheet:
            col0 = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=str).iloc[:, 0]
            col0 = col0.dropna().astype(str).map(_norm_ws)
            col0 = col0[col0 != ""]
            text_questions[tag_from_sheet] = col0.tolist()
            # Also look for additional placements in same sheet
            for k, v in harvest_text_columns(df_raw).items():
                text_questions[k if k not in text_questions else f"{k}__sheet:{sheet}"] = v
            for k, v in harvest_text_anywhere(df_raw).items():
                if k not in text_questions:
                    text_questions[k] = v
            continue

        # 2) numeric parsing
        tidy_sheet, per_q_overall = parse_crosstab_sheet(df_raw)
        if not tidy_sheet.empty:
            all_rows.append(tidy_sheet)
            overall_stats.update(per_q_overall)

        # 3) harvest _text columns + anywhere inside numeric sheets
        for k, v in harvest_text_columns(df_raw).items():
            text_questions[k if k not in text_questions else f"{k}__sheet:{sheet}"] = v
        for k, v in harvest_text_anywhere(df_raw).items():
            if k not in text_questions:
                text_questions[k] = v

    tidy_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=["question","response","demographic","count","percent"])
    return tidy_df, overall_stats, text_questions

# -------------------------
# Fallback keyword summarizer (if Ollama unavailable)
# -------------------------
def summarize_text_list(texts: List[str], top_k: int = 12):
    tokens = []
    for t in texts:
        for w in str(t).lower().split():
            w = "".join(ch for ch in w if ch.isalnum() or ch in "-'")
            if not w or w in STOPWORDS or w.isnumeric():
                continue
            tokens.append(w)
    common = Counter(tokens).most_common(top_k)
    bullets = [f"- Frequent theme: **{w}**" for w, _ in common]
    return bullets, pd.DataFrame(common, columns=["term", "count"])

def question_sort_key(q: str):
    m = re.search(r"[SA](\d+)Q(\d+)", q, re.IGNORECASE)
    if m:
        return (int(m.group(1)), int(m.group(2)), q)
    return (10**9, 10**9, q.lower())

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Survey Crosstab Visualizer", layout="wide")
st.title("📊 Survey Crosstab Visualizer")
st.caption("Upload a crosstab as **.xlsx** (any number of sheets) or **.csv** (single grid).")

@st.cache_data(show_spinner=False)
def _parse_all(file_bytes: bytes, filename: str):
    bio = io.BytesIO(file_bytes)
    bio.name = filename
    tidy_df, overall_stats, text_questions = parse_workbook(bio)
    return tidy_df, overall_stats, text_questions

uploaded = st.file_uploader("Upload .xlsx or .csv", type=["xlsx", "csv"])
if not uploaded:
    st.info("Upload your crosstab to begin.")
    st.stop()

with st.spinner("Parsing…"):
    tidy_df, overall_stats, text_questions = _parse_all(uploaded.getvalue(), uploaded.name)

with st.expander("Debug: detected _text questions", expanded=False):
    st.write(sorted(list(text_questions.keys())))

has_numeric = not tidy_df.empty
has_text = len(text_questions) > 0
if not has_numeric and not has_text:
    st.warning("No recognizable crosstab blocks or `_text` questions were found.")
    st.stop()

st.sidebar.header("Controls")
mode = st.sidebar.radio("View", ["Graph questions", "Summarize _text questions"],
                        index=0 if has_numeric else 1)

# -------------------------
# GRAPH MODE (numeric)
# -------------------------
if mode == "Graph questions":
    all_questions = sorted(tidy_df["question"].unique(), key=question_sort_key)
    search = st.sidebar.text_input("Search question", "")
    opts = [q for q in all_questions if search.lower() in q.lower()] or all_questions
    q = st.sidebar.selectbox("Question", options=opts)

    if _is_text_question(q):
        st.info("This question is marked as open-ended (`_text`). Switch to **Summarize _text questions** in the sidebar.")
        st.stop()

    qdf = tidy_df[tidy_df["question"] == q].copy()

    demogs = sorted(qdf["demographic"].unique())
    sel_demogs = st.sidebar.multiselect("Demographics", demogs, default=demogs)
    qdf = qdf[qdf["demographic"].isin(sel_demogs)]

    responses = list(qdf["response"].unique())
    sel_responses = st.sidebar.multiselect("Responses", responses, default=responses)
    qdf = qdf[qdf["response"].isin(sel_responses)]

    metric = st.sidebar.radio("Metric to display", ["Percent", "Count"], index=0)
    yvar = "percent" if metric == "Percent" else "count"

    chart_type = st.sidebar.selectbox("Chart type", ["Grouped Bar", "Stacked Bar", "100% Stacked Bar", "Heatmap"])
    normalize_toggle = False
    if chart_type == "Stacked Bar" and metric == "Percent":
        normalize_toggle = st.sidebar.checkbox("Normalize stacks to 100% (by response)", value=False)
    force_100 = st.sidebar.checkbox("Force Y-axis [0,100] (Percent only)", value=(metric == "Percent" and "Stacked" in chart_type))
    show_values = st.sidebar.checkbox("Show value labels", True)

    if qdf.empty:
        st.warning("No data to display based on your filters.")
        st.stop()

    # EXCLUDE "Total" from charts
    qdf_no_total = qdf[qdf["demographic"] != "Total"]

    sort_by = st.sidebar.selectbox("Sort responses by", ["original", "alphabetical", "mean of selected metric (desc)"])
    if sort_by == "alphabetical":
        ordered = sorted(sel_responses)
    elif sort_by == "mean of selected metric (desc)":
        ordered = list(qdf_no_total.groupby("response")[yvar].mean().sort_values(ascending=False).index)
    else:
        ordered = sel_responses

    plot_df = qdf.copy()  # includes "Total" for preview/export
    plot_df["response"] = pd.Categorical(plot_df["response"], categories=ordered, ordered=True)
    plot_df = plot_df.sort_values(["response", "demographic"])

    plot_df_chart = plot_df[plot_df["demographic"] != "Total"].copy()
    effective_chart, using_norm = chart_type, False
    if chart_type == "100% Stacked Bar":
        if metric != "Percent":
            effective_chart = "Stacked Bar"
        else:
            plot_df_chart = _normalize_by_response(plot_df_chart); using_norm = True
    if chart_type == "Stacked Bar" and metric == "Percent" and normalize_toggle:
        plot_df_chart = _normalize_by_response(plot_df_chart); using_norm = True; effective_chart = "100% Stacked Bar"

    left, right = st.columns([2.3, 1])
    with left:
        st.subheader(q if len(q) < 160 else q[:157] + "…")
        if q in overall_stats and overall_stats[q] is not None:
            st.caption(f"Overall Stat Test of Percentages: **{overall_stats[q]:.3f}**")

        if effective_chart in ["Grouped Bar", "Stacked Bar", "100% Stacked Bar"]:
            barmode = "group" if effective_chart == "Grouped Bar" else "stack"
            y_to_plot = "norm_percent" if (using_norm and metric == "Percent") else yvar
            y_label   = "Percent" if y_to_plot in ["percent", "norm_percent"] else "Count"
            fig = px.bar(plot_df_chart, x="response", y=y_to_plot, color="demographic", barmode=barmode,
                         labels={y_to_plot: y_label, "response": "Response"})
        else:
            piv = plot_df_chart.pivot_table(index="response", columns="demographic",
                                            values=("norm_percent" if (using_norm and metric == "Percent") else yvar),
                                            aggfunc="mean")
            fig = px.imshow(piv, aspect="auto", origin="lower",
                            labels=dict(x="Demographic", y="Response",
                                        color=("Percent" if metric == "Percent" else "Count")))

        if effective_chart != "Heatmap":
            if metric == "Percent":
                if using_norm or force_100:
                    fig.update_yaxes(range=[0, 100], autorange=False, rangemode="tozero")
                else:
                    fig.update_yaxes(autorange=True, rangemode="tozero")
            else:
                fig.update_yaxes(autorange=True, rangemode="tozero")

        if show_values and effective_chart != "Heatmap":
            fig.update_traces(texttemplate=("%{y:.1f}" if metric == "Percent" else "%{y:.0f}"),
                              textposition="outside", selector=dict(type="bar"))
        fig.update_layout(title=None, title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Downloads**")
        st.download_button("⬇️ Data (CSV)",
                           plot_df.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_data.csv", mime="text/csv")

    with right:
        st.subheader("Data preview (includes 'Total')")
        st.dataframe(plot_df, use_container_width=True, height=520)

# -------------------------
# TEXT MODE (open-ended via Ollama)
# -------------------------
else:
    if not text_questions:
        st.info("No `_text` questions detected (sheet names, column headers, or in-sheet cells).")
        st.stop()

    tq = st.sidebar.selectbox("Open-ended question", sorted(text_questions.keys()))
    responses_raw = text_questions[tq]
    st.subheader(f"Open-ended responses for: {tq}")

    with st.expander("LLM summarization settings (Ollama)", expanded=True):
        use_ollama = st.checkbox("Use Ollama to summarize (recommended)", value=True)
        model = st.text_input("Ollama model", "llama3.1:8b")
        host = st.text_input("Ollama host", "http://localhost:11434")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        max_tokens = st.slider("Max tokens (approx.)", 200, 1200, 600, 50)

    q = st.text_input("Filter keyword (preview only)", "")
    shown = [r for r in responses_raw if q.lower() in r.lower()]
    st.write(f"Showing {len(shown)} of {len(responses_raw)} responses")
    st.dataframe(pd.DataFrame({"response": shown}), use_container_width=True, height=220)
    st.download_button("⬇️ Responses (CSV)",
                       pd.DataFrame({"response": responses_raw}).to_csv(index=False).encode("utf-8"),
                       file_name=f"{tq}_responses.csv", mime="text/csv")

    if use_ollama:
        with st.spinner("Summarizing with Ollama…"):
            summary = ollama_summarize(responses_raw, model=model, host=host,
                                       max_tokens=max_tokens, temperature=temperature)
        st.markdown("### Confidential summary (LLM)")
        st.markdown(summary)
        if summary.startswith("Could not reach Ollama"):
            st.info("Falling back to keyword themes below.")
    else:
        st.info("Ollama disabled — showing keyword themes instead.")

    st.markdown("### Quick keyword themes (fallback)")
    bullets, top = summarize_text_list(responses_raw, top_k=20)
    st.markdown("\n".join(bullets))
    st.dataframe(top, use_container_width=True, height=260)
