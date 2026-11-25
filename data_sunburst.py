from pathlib import Path
import textwrap
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Findmypast Dataset Viewer",
    layout="wide",
)

# Prevent tall hover cards from clipping at the bottom edge of the chart container
st.markdown(
    """
    <style>
    .js-plotly-plot .plotly .main-svg,
    .stPlotlyChart > div {
        overflow: visible !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Configuration ---
# --- Configuration ---
DATA_PATH = Path("df_output_for_donut.parquet")  # Also supports .pkl or .csv
DATA_AS_OF = "25 Nov 2026"
BASE_URL = "https://search.findmypast.co.uk/search-world-records/"
TOP_N_CATEGORIES = 12
BOTTOM_N_CATEGORIES = 24

#DATA_PATH = Path("df_output_for_donut_0924.csv")  # Also supports .parquet or .pkl
#DATA_AS_OF = "Sept 2024"
#BASE_URL = "https://search.findmypast.co.uk/search-world-records/"
#TOP_N_CATEGORIES = 12

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


@st.cache_data
def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
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


master, cat_totals, all_countries = load_data(DATA_PATH)
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

st.sidebar.divider()

cat_range = st.sidebar.slider(
    "Whole category items (m)",
    min_value=0.0,
    max_value=1200.0,
    value=(0.0, 1200.0),
    step=0.5,
)

rec_range = st.sidebar.slider(
    "Recordset items (m)",
    min_value=0.0,
    max_value=300.0,
    value=(0.0, 300.0),
    step=0.5,
)

available_types = sorted(master["recordtype"].unique().tolist())
selected_types = st.sidebar.multiselect(
    "Record types",
    options=available_types,
    default=available_types,
)

# --- Apply Filters ---
if cat_select == "ALL":
    valid_cats = cat_totals.loc[
        (cat_totals["events"] >= cat_range[0] * 1_000_000)
        & (cat_totals["events"] <= cat_range[1] * 1_000_000),
        "cat_title",
    ]
    cat_mask = master["cat_title"].isin(valid_cats)
else:
    cat_mask = master["cat_title"] == cat_select

rec_mask = (master["events"] >= rec_range[0] * 1_000_000) & (master["events"] <= rec_range[1] * 1_000_000)
type_mask = master["recordtype"].isin(selected_types)

if selected_countries:
    country_set = set(selected_countries)
    country_mask = master["source_country_list"].str.split(", ").apply(lambda lst: bool(country_set.intersection(lst)))
else:
    country_mask = pd.Series(False, index=master.index)

# Build colour map from the un-zoomed category order so zooming in preserves each category's colour
overview_valid_cats = cat_totals.loc[
    (cat_totals["events"] >= cat_range[0] * 1_000_000)
    & (cat_totals["events"] <= cat_range[1] * 1_000_000),
    "cat_title",
]
overview_df = master.loc[
    master["cat_title"].isin(overview_valid_cats) & rec_mask & type_mask & country_mask
]
overview_order = (
    overview_df.groupby("cat_title")["events"]
    .sum()
    .sort_values(ascending=False)
    .index.tolist()
)
if view_slice == f"Top {TOP_N_CATEGORIES}":
    overview_order = overview_order[:TOP_N_CATEGORIES]
elif view_slice == "Long Tail":
    overview_order = overview_order[TOP_N_CATEGORIES:]

color_map = {
    cat: BOKEH_CATEGORY20B[i % len(BOKEH_CATEGORY20B)]
    for i, cat in enumerate(overview_order)
}
# Fallback in case a category is picked via dropdown while excluded by cat_range/view_slice
global_order = cat_totals.sort_values("events", ascending=False)["cat_title"].tolist()
for i, cat in enumerate(global_order):
    color_map.setdefault(cat, BOKEH_CATEGORY20B[i % len(BOKEH_CATEGORY20B)])

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
st.title(f"Findmypast Datasets ({DATA_AS_OF})")

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
            marker=dict(line=dict(width=0.5, color="white")),
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