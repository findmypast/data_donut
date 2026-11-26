from datetime import datetime
from pathlib import Path
import re
import textwrap
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Findmypast Dataset Viewer",
    layout="wide",
)

# Prevent tall hover cards from clipping at the bottom edge of the chart container
# and keep sidebar reset buttons from wrapping/truncating on narrow widths
st.markdown(
    """
    <style>
    .js-plotly-plot .plotly .main-svg,
    .stPlotlyChart > div {
        overflow: visible !important;
    }
    section[data-testid="stSidebar"] button {
        white-space: nowrap !important;
        padding-left: 0.45rem !important;
        padding-right: 0.45rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input {
        font-size: 0.84rem !important;
        padding-left: 0.45rem !important;
        padding-right: 0.25rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button {
        padding-left: 0.25rem !important;
        padding-right: 0.25rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Configuration ---
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
LEGACY_DATA_PATH = APP_DIR / "df_output_for_donut.parquet"
DATE_SUFFIX_RE = re.compile(r"^df_output_for_donut_(\d{4}-\d{2}-\d{2})\.parquet$")
REQUIRED_COLS = {
    "dataset",
    "dataset_orig",
    "category",
    "recordtype",
    "events",
    "source_country_list",
    "country_events_contrib",
    "rmid_events_contrib",
}

BASE_URL = "https://search.findmypast.co.uk/search-world-records/"
TOP_N_CATEGORIES = 12
BOTTOM_N_CATEGORIES = 24

# Exact even-indexed colours from Bokeh's Category20b[20]
BOKEH_CATEGORY20B = [
    "#393b79",  # dark indigo
    "#6b6ecf",  # periwinkle
    "#637939",  # olive green
    "#b5cf6b",  # sage
    "#8c6d31",  # ochre
    "#e7ba52",  # gold
    "#843c39",  # brick red
    "#d6616b",  # rose
    "#7b4173",  # plum
    "#ce6dbd",  # orchid
]

COUNTRY_PRESETS = {
    "ALL": None,
    "UK": ["England", "Great Britain", "Scotland", "UK None", "UK Other", "United Kingdom", "Wales"],
    "UK & Ireland": ["England", "Great Britain", "Ireland", "Scotland", "UK None", "UK Other", "United Kingdom", "Wales"],
    "Ireland": ["Ireland"],
    "Americas": ["Americas", "Canada", "Caribbean", "Central America", "North America", "United States"],
    "Australia & NZ": ["Australasia", "Australia", "New Zealand"],
    "Asia": ["Asia", "India"],
}


def format_events(val: float) -> str:
    m = val / 1_000_000
    if m > 10:
        return f"{m:,.1f} m"
    if m > 1:
        return f"{m:,.2f} m"
    return f"{m:,.3f} m"


def wrap_hover_text(val: str, width: int = 60, max_lines: int = 5) -> str:
    """Wrap comma-separated lists onto <br>-delimited lines so tooltips stay compact."""
    if pd.isna(val) or not str(val).strip():
        return "None"
    lines = textwrap.wrap(str(val), width=width, break_long_words=False)
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["… (see Dataset Table tab for full list)"]
    return "<br>".join(lines)


def format_date_from_path(parquet_path: Path) -> str:
    """Extract YYYY-MM-DD from filename suffix and format as e.g. '26 Nov 2026'."""
    match = DATE_SUFFIX_RE.match(parquet_path.name)
    if match:
        dt = datetime.strptime(match.group(1), "%Y-%m-%d")
        return dt.strftime("%-d %b %Y")
    mtime = datetime.fromtimestamp(parquet_path.stat().st_mtime)
    return mtime.strftime("%-d %b %Y")


def resolve_latest_valid_dataset() -> tuple[Path, str]:
    """
    Find the newest valid/complete Parquet file in ./data/ (or legacy root path).
    Skips empty, missing-column, or corrupt files so a failed refresh never breaks the app.
    """
    candidates: list[tuple[str, Path]] = []

    if DATA_DIR.exists():
        for p in DATA_DIR.glob("df_output_for_donut_*.parquet"):
            m = DATE_SUFFIX_RE.match(p.name)
            if m:
                candidates.append((m.group(1), p))

    candidates.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    ordered_paths = [p for _, p in candidates]

    if LEGACY_DATA_PATH.exists():
        ordered_paths.append(LEGACY_DATA_PATH)

    for candidate in ordered_paths:
        try:
            if not candidate.is_file() or candidate.stat().st_size == 0:
                continue
            probe = pd.read_parquet(candidate)
            if probe.empty:
                continue
            if not REQUIRED_COLS.issubset(set(probe.columns)):
                continue
            return candidate, format_date_from_path(candidate)
        except Exception:
            continue

    raise FileNotFoundError(
        f"No valid df_output_for_donut Parquet file found in {DATA_DIR} or {LEGACY_DATA_PATH}."
    )


@st.cache_data(show_spinner=False)
def load_data(path_str: str, mtime_ns: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    path = Path(path_str)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".pkl":
        df = pd.read_pickle(path)
    else:
        df = pd.read_csv(path)

    df = df.copy()

    # Titles & display strings
    df["cat_title"] = df["category"].astype(str).str.title()
    df["dataset_title"] = (
        df["dataset"]
        .astype(str)
        .str.title()
        .str.replace("Us", "US", regex=False)
        .str.replace("Uk", "UK", regex=False)
        .str.replace("'S", "'s", regex=False)
    )
    df["recordtype"] = df["recordtype"].astype(str).str.title()
    df["source_country_list"] = (
        df["source_country_list"]
        .fillna("None")
        .astype(str)
        .str.title()
        .str.replace("Uk", "UK", regex=False)
    )
    df["str_from_events"] = df["events"].map(format_events)
    df["country_contrib_wrapped"] = df["country_events_contrib"].map(wrap_hover_text)
    df["rmid_contrib_wrapped"] = df["rmid_events_contrib"].map(wrap_hover_text)

    # Search URL slug
    slug = (
        df["dataset_orig"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("&", "and", regex=False)
        .str.replace(" ", "-", regex=False)
        .str.replace(r"vols\.", "vols", regex=True)
        .str.replace(r"vol\.", "vol", regex=True)
    )
    df["search_url"] = BASE_URL + slug

    cat_totals = df.groupby("cat_title", as_index=False)["events"].sum()
    countries = sorted({c for row in df["source_country_list"].str.split(", ") for c in row})
    return df, cat_totals, countries


def nice_ceil_millions(max_val_m: float, min_step: float = 0.0001) -> float:
    """Round an upper bound in millions up to a clean step for the current filter context."""
    if max_val_m <= 0:
        return max(min_step, 0.0001)
    if max_val_m <= 0.005:
        step = max(min_step, 0.0001)
    elif max_val_m <= 0.05:
        step = max(min_step, 0.001)
    elif max_val_m <= 0.5:
        step = max(min_step, 0.01)
    elif max_val_m <= 2.0:
        step = max(min_step, 0.05)
    elif max_val_m <= 10.0:
        step = max(min_step, 0.5)
    elif max_val_m <= 100.0:
        step = max(min_step, 5.0)
    elif max_val_m <= 500.0:
        step = max(min_step, 25.0)
    else:
        step = max(min_step, 50.0)
    return round(float(np.ceil(max_val_m / step) * step), 5)


def render_min_max_filter(
    label: str,
    prefix: str,
    context_max_m: float,
    step: float,
    fmt: str,
) -> tuple[float, float]:
    """
    Context-aware Min & Max number inputs in millions with a Reset button and
    automatic clamping to [0.0, max_bound].
    """
    max_bound = nice_ceil_millions(context_max_m, min_step=step)
    min_key = f"{prefix}_min_m"
    max_key = f"{prefix}_max_m"
    bound_key = f"{prefix}_last_max_bound"

    if min_key not in st.session_state:
        st.session_state[min_key] = 0.0
    if max_key not in st.session_state:
        st.session_state[max_key] = max_bound

    # Autocorrect out-of-context values whenever the context ceiling changes
    prev_bound = st.session_state.get(bound_key)
    if prev_bound is not None and abs(prev_bound - max_bound) > 1e-9:
        if abs(float(st.session_state[max_key]) - prev_bound) < 1e-9:
            st.session_state[max_key] = max_bound
        else:
            st.session_state[max_key] = max(0.0, min(float(st.session_state[max_key]), max_bound))
        st.session_state[min_key] = max(0.0, min(float(st.session_state[min_key]), float(st.session_state[max_key])))
    else:
        st.session_state[max_key] = max(0.0, min(float(st.session_state[max_key]), max_bound))
        st.session_state[min_key] = max(0.0, min(float(st.session_state[min_key]), float(st.session_state[max_key])))

    st.session_state[bound_key] = max_bound

    def _on_num_change() -> None:
        cur_max = st.session_state[bound_key]
        lo_val = max(0.0, min(cur_max, float(st.session_state[min_key])))
        hi_val = max(0.0, min(cur_max, float(st.session_state[max_key])))
        if lo_val > hi_val:
            hi_val = lo_val
        st.session_state[min_key] = round(lo_val, 5)
        st.session_state[max_key] = round(hi_val, 5)

    def _on_reset() -> None:
        st.session_state[min_key] = 0.0
        st.session_state[max_key] = st.session_state[bound_key]

    hdr_left, hdr_right = st.columns([2.2, 1])
    with hdr_left:
        st.markdown(
            f"**{label}**<br><span style='font-size:0.85em;font-style:italic;'>max {max_bound:g}m</span>",
            unsafe_allow_html=True,
        )
    with hdr_right:
        st.button(
            "↺ Reset",
            key=f"{prefix}_reset_btn",
            on_click=_on_reset,
            help="Reset Min to 0 and Max to current context max",
            use_container_width=True,
        )

    c_min, c_max = st.columns(2)
    with c_min:
        st.number_input(
            "Min (m)",
            min_value=0.0,
            max_value=max_bound,
            step=step,
            format=fmt,
            key=min_key,
            on_change=_on_num_change,
        )
    with c_max:
        st.number_input(
            "Max (m)",
            min_value=0.0,
            max_value=max_bound,
            step=step,
            format=fmt,
            key=max_key,
            on_change=_on_num_change,
        )

    return float(st.session_state[min_key]), float(st.session_state[max_key])


data_path, data_as_of = resolve_latest_valid_dataset()
master, cat_totals, all_countries = load_data(str(data_path), data_path.stat().st_mtime_ns)
cat_options = ["ALL"] + sorted(cat_totals["cat_title"].tolist())
valid_cat_set = set(cat_totals["cat_title"].tolist())

# --- Shared Zoom / Selection State (applied before sidebar widget renders) ---
if "cat_select" not in st.session_state:
    st.session_state["cat_select"] = "ALL"
if "chart_rev" not in st.session_state:
    st.session_state["chart_rev"] = 0
if "pending_cat_select" in st.session_state:
    st.session_state["cat_select"] = st.session_state.pop("pending_cat_select")

# --- Sidebar Controls ---
st.sidebar.header("Filters")

cat_select = st.sidebar.selectbox("Category view", cat_options, key="cat_select")

view_slice = st.sidebar.radio(
    "Category slice (zoom long tail)",
    options=["All", f"Top {TOP_N_CATEGORIES}", "Long Tail", f"Bottom {BOTTOM_N_CATEGORIES}"],
    horizontal=True,
    disabled=(cat_select != "ALL"),
    help="Isolate the largest categories, expand the long tail, or zoom into the smallest categories.",
)

preset = st.sidebar.selectbox("Country quick selection", list(COUNTRY_PRESETS.keys()), index=0)
default_countries = all_countries if COUNTRY_PRESETS[preset] is None else [
    c for c in COUNTRY_PRESETS[preset] if c in all_countries
]
selected_countries = st.sidebar.multiselect(
    "Source country contains",
    options=all_countries,
    default=default_countries,
)

available_types = sorted(master["recordtype"].unique().tolist())
selected_types = st.sidebar.multiselect(
    "Record types",
    options=available_types,
    default=available_types,
)

st.sidebar.divider()

# Base context from active country and record type only (stable against range & focus changes)
type_mask = master["recordtype"].isin(selected_types)
if selected_countries:
    country_set = set(selected_countries)
    country_mask = master["source_country_list"].str.split(", ").apply(lambda lst: bool(country_set.intersection(lst)))
else:
    country_mask = pd.Series(False, index=master.index)

base_context_df = master.loc[type_mask & country_mask]
base_cat_sums = base_context_df.groupby("cat_title")["events"].sum()

# Fixed category order for colours within the current country + record-type selection
stable_cat_order = base_cat_sums.sort_values(ascending=False).index.tolist()
color_map = {
    cat: BOKEH_CATEGORY20B[i % len(BOKEH_CATEGORY20B)]
    for i, cat in enumerate(stable_cat_order)
}
global_order = cat_totals.sort_values("events", ascending=False)["cat_title"].tolist()
for i, cat in enumerate(global_order):
    color_map.setdefault(cat, BOKEH_CATEGORY20B[i % len(BOKEH_CATEGORY20B)])

# Context bounds for the min/max inputs (respects single-category focus and slice)
bounds_df = base_context_df[base_context_df["cat_title"] == cat_select] if cat_select != "ALL" else base_context_df
bounds_cat_sums = bounds_df.groupby("cat_title")["events"].sum()

if cat_select == "ALL" and not bounds_cat_sums.empty and view_slice != "All":
    ranked_ctx_cats = bounds_cat_sums.sort_values(ascending=False).index.tolist()
    if view_slice == f"Top {TOP_N_CATEGORIES}":
        ctx_keep = set(ranked_ctx_cats[:TOP_N_CATEGORIES])
    elif view_slice == "Long Tail":
        ctx_keep = set(ranked_ctx_cats[TOP_N_CATEGORIES:])
    else:  # Bottom N
        ctx_keep = set(ranked_ctx_cats[-BOTTOM_N_CATEGORIES:])
    bounds_cat_sums = bounds_cat_sums[bounds_cat_sums.index.isin(ctx_keep)]
    bounds_df = bounds_df[bounds_df["cat_title"].isin(ctx_keep)]

cat_context_max_m = (float(bounds_cat_sums.max()) / 1_000_000) if not bounds_cat_sums.empty else 0.01
rec_context_max_m = (float(bounds_df["events"].max()) / 1_000_000) if not bounds_df.empty else 0.001

with st.sidebar:
    cat_range = render_min_max_filter(
        label="Whole category items (m)",
        prefix="cat_flt",
        context_max_m=cat_context_max_m,
        step=0.01,
        fmt="%.2f",
    )
    st.markdown("")
    rec_range = render_min_max_filter(
        label="Recordset items (m)",
        prefix="rec_flt",
        context_max_m=rec_context_max_m,
        step=0.0001,
        fmt="%.4f",
    )

# --- Apply Filters ---
rec_mask = (master["events"] >= rec_range[0] * 1_000_000) & (master["events"] <= rec_range[1] * 1_000_000)

if cat_select == "ALL":
    valid_cats = base_cat_sums[
        (base_cat_sums >= cat_range[0] * 1_000_000)
        & (base_cat_sums <= cat_range[1] * 1_000_000)
    ].index
    cat_mask = master["cat_title"].isin(valid_cats)
else:
    cat_mask = master["cat_title"] == cat_select

filtered = master.loc[cat_mask & rec_mask & type_mask & country_mask].copy()

# Apply Top N / Long Tail / Bottom N slice after other filters
if cat_select == "ALL" and not filtered.empty and view_slice != "All":
    ranked_cats = (
        filtered.groupby("cat_title")["events"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    if view_slice == f"Top {TOP_N_CATEGORIES}":
        keep_cats = set(ranked_cats[:TOP_N_CATEGORIES])
    elif view_slice == "Long Tail":
        keep_cats = set(ranked_cats[TOP_N_CATEGORIES:])
    elif view_slice == f"Bottom {BOTTOM_N_CATEGORIES}":
        keep_cats = set(ranked_cats[-BOTTOM_N_CATEGORIES:])
    filtered = filtered[filtered["cat_title"].isin(keep_cats)].copy()

# --- Header & KPIs ---
st.title(f"Findmypast Datasets ({data_as_of})")

c1, c2, c3 = st.columns(3)
c1.metric("Items", f"{filtered['events'].sum():,}")
c2.metric("Datasets", f"{len(filtered):,}")
c3.metric("Categories", f"{filtered['cat_title'].nunique():,}")

if filtered.empty:
    st.warning("No datasets match the current filter selection.")
    st.stop()

# --- Visualisation & Table Tabs ---
tab_chart, tab_table = st.tabs(["Chart", "Dataset Table & Links"])

with tab_chart:
    ctrl_left, ctrl_right = st.columns([3, 1])
    with ctrl_left:
        chart_type = st.radio(
            "Chart type",
            options=["Sunburst", "Treemap"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with ctrl_right:
        if cat_select != "ALL":
            if st.button("← Back to All Categories", width="stretch"):
                st.session_state["pending_cat_select"] = "ALL"
                st.session_state["chart_rev"] += 1
                st.rerun()

    cat_hover_tmpl = (
        "<b>%{label}</b><br>"
        "Items: %{value:,}<br>"
        "Category: %{label}"
        "<extra></extra>"
    )

    dataset_hover_tmpl = (
        "<b>%{label}</b><br>"
        "Items: %{value:,}<br>"
        "Category: %{customdata[1]}<br>"
        "Type: %{customdata[2]}<br>"
        "<br><b>Source countries:</b><br>%{customdata[3]}<br>"
        "<br><b>rmid contributions:</b><br>%{customdata[4]}"
        "<extra></extra>"
    )

    common_kwargs = dict(
        data_frame=filtered,
        path=["cat_title", "dataset_title"],
        values="events",
        color="cat_title",
        color_discrete_map=color_map,
        custom_data=[
            "str_from_events",
            "cat_title",
            "recordtype",
            "country_contrib_wrapped",
            "rmid_contrib_wrapped",
        ],
    )

    if chart_type == "Sunburst":
        fig = px.sunburst(**common_kwargs)
        fig.update_traces(
            sort=True,
            rotation=90,
            leaf=dict(opacity=0.78),
            marker=dict(line=dict(width=0)),
            insidetextorientation="radial",
        )
        tip_text = "Tip: Click any category slice to focus the chart and table on that category; click the centre circle or 'Back to All Categories' to zoom back out."
    else:
        fig = px.treemap(**common_kwargs)
        fig.update_traces(
            sort=True,
            marker=dict(line=dict(width=1, color="white")),
        )
        tip_text = "Tip: Click any category header or tile to focus the chart and table on that category; click the category header or 'Back to All Categories' to zoom back out."

    # Short 3-line card for top-level category nodes; full card for dataset leaf nodes
    fig.data[0].hovertemplate = [
        cat_hover_tmpl if parent == "" else dataset_hover_tmpl
        for parent in fig.data[0].parents
    ]

    fig.update_layout(
        height=860,
        margin=dict(t=25 if chart_type == "Treemap" else 10, l=10, r=10, b=120),
        hoverlabel=dict(align="left"),
    )

    event = st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key=f"chart_{chart_type}_{st.session_state['chart_rev']}",
    )
    st.caption(tip_text)

    # Sync click-to-zoom with Category view, other chart type, KPIs, and Dataset Table
    selected_points = event.selection.points if event and event.selection else []
    if selected_points:
        pt = selected_points[0]
        label = pt.get("label", "")
        parent = pt.get("parent") or ""

        if cat_select == "ALL":
            target_cat = label if parent == "" else parent
            if target_cat in valid_cat_set:
                st.session_state["pending_cat_select"] = target_cat
                st.session_state["chart_rev"] += 1
                st.rerun()
        elif parent == "":
            # Clicked the top-level category node while already zoomed in -> zoom back out
            st.session_state["pending_cat_select"] = "ALL"
            st.session_state["chart_rev"] += 1
            st.rerun()

with tab_table:
    display_df = filtered[[
        "dataset_title",
        "cat_title",
        "recordtype",
        "events",
        "source_country_list",
        "search_url",
    ]].sort_values("events", ascending=False)

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "dataset_title": "Dataset",
            "cat_title": "Category",
            "recordtype": "Type",
            "events": st.column_config.NumberColumn("Items", format="%d"),
            "source_country_list": "Source Countries",
            "search_url": st.column_config.LinkColumn("Findmypast Link", display_text="Open"),
        },
    )
