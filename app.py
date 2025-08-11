# =========================
# USA Housing (King County) Dashboard
# =========================
# How to run:
#   pip install streamlit pandas numpy plotly statsmodels
#   streamlit run app.py
# Place your CSV (e.g., "USA Housing Dataset.csv" or "kc_house_data.csv")
# in the same folder as this script, or upload it from the sidebar.

import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="USA Housing Prices — Interactive Dashboard",
    layout="wide",
    page_icon="🏠",
)

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def read_csv_safely(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]
    return df

def find_col(df: pd.DataFrame, keys, must=False):
    """Find first column whose name contains ALL tokens in `keys`."""
    cols = df.columns.tolist()
    keys = [k.lower() for k in ([keys] if isinstance(keys, str) else keys)]
    for c in cols:
        name = c.lower()
        if all(k in name for k in keys):
            return c
    if must:
        raise KeyError(f"Required column not found: {keys}")
    return None

def add_derived_columns(df: pd.DataFrame, price_col, sqft_col, yr_built_col, yr_renov_col, statezip_col):
    df = df.copy()

    # price per sqft
    if price_col and sqft_col and (sqft_col in df) and (price_col in df):
        with np.errstate(divide='ignore', invalid='ignore'):
            df["price_per_sqft"] = np.where(df[sqft_col] > 0, df[price_col] / df[sqft_col], np.nan)

    # renovated flag
    if yr_renov_col and yr_renov_col in df:
        df["renovated_flag"] = df[yr_renov_col].fillna(0).astype(int) > 0

    # parse state / zip from statezip like "WA 98052"
    if statezip_col and statezip_col in df:
        s = df[statezip_col].astype(str)
        df["state"] = s.str.extract(r"^([A-Za-z]{2})")[0]
        df["zip"] = s.str.extract(r"(\d{5})")[0]
    else:
        # try to create "state" if explicit col exists
        if "state" not in df.columns:
            # Many King County sets only have WA, so fill WA if city exists
            if "city" in df.columns:
                df["state"] = "WA"

    # clean city formatting
    if "city" in df.columns:
        df["city"] = df["city"].astype(str).str.strip().str.title()

    return df

def kpi_card(col, label, value, prefix="", suffix=""):
    if pd.isna(value):
        txt = "—"
    else:
        if isinstance(value, (int, np.integer)):
            txt = f"{prefix}{value:,}{suffix}"
        elif isinstance(value, (float, np.floating)):
            if "price" in label.lower():
                txt = f"{prefix}{value:,.0f}{suffix}"
            else:
                txt = f"{prefix}{value:,.1f}{suffix}"
        else:
            txt = f"{value}"
    col.metric(label, txt)

# -------------------------
# Load Data
# -------------------------
st.sidebar.header("Upload CSV (optional)")
uploaded = st.sidebar.file_uploader("Drag & drop or browse", type=["csv"], label_visibility="collapsed")

df = None
default_files = ["USA Housing Dataset.csv", "kc_house_data.csv", "kc_house_data.csv.gz"]

if uploaded:
    df = pd.read_csv(uploaded)
else:
    for f in default_files:
        if os.path.exists(f):
            df = read_csv_safely(f)
            break

if df is None:
    st.info("📄 Please upload a CSV file to get started.")
    st.stop()

df = normalize_cols(df)

# -------------------------
# Column detection (robust to different naming)
# -------------------------
price_col     = find_col(df, ["price"], must=True)
bed_col       = find_col(df, ["bed"]) or find_col(df, ["bedroom"])
bath_col      = find_col(df, ["bath"]) or find_col(df, ["bathroom"])
sqft_living   = find_col(df, ["sqft", "living"]) or find_col(df, ["living", "area"])
sqft_lot      = find_col(df, ["sqft", "lot"])
floors_col    = find_col(df, ["floor"])
waterfront    = find_col(df, ["waterfront"])
view_col      = find_col(df, ["view"])
condition_col = find_col(df, ["condition"])
grade_col     = find_col(df, ["grade"])
sqft_above    = find_col(df, ["sqft", "above"])
sqft_basement = find_col(df, ["sqft", "basement"])
yr_built_col  = find_col(df, ["yr", "built"]) or find_col(df, ["year_built"])
yr_renov_col  = find_col(df, ["yr", "renov"]) or find_col(df, ["year_renov"])
street_col    = find_col(df, ["street"])
city_col      = find_col(df, ["city"])
statezip_col  = find_col(df, ["statezip"]) or find_col(df, ["state", "zip"])
state_col     = "state" if "state" in df.columns else None
lat_col       = find_col(df, ["lat"])
long_col      = find_col(df, ["long"]) or find_col(df, ["lng"])

# Derived columns
df = add_derived_columns(df, price_col, sqft_living, yr_built_col, yr_renov_col, statezip_col)
state_col = "state" if "state" in df.columns else state_col

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.header("Filters")

# State filter (many King County files are WA only)
if state_col and state_col in df.columns:
    state_opts = sorted([x for x in df[state_col].dropna().unique().tolist()])
    selected_states = st.sidebar.multiselect("Select State(s)", state_opts, default=state_opts)
    if selected_states:
        df = df[df[state_col].isin(selected_states)]

# City filter
if city_col and city_col in df.columns:
    city_opts = df[city_col].dropna().unique().tolist()
    city_opts = sorted(city_opts)
    default_cities = city_opts[:20] if len(city_opts) > 20 else city_opts
    selected_cities = st.sidebar.multiselect("Select City(s)", city_opts, default=default_cities)
    if selected_cities:
        df = df[df[city_col].isin(selected_cities)]

# Price range
pmin, pmax = float(df[price_col].min()), float(df[price_col].max())
sel_price = st.sidebar.slider("Price range", pmin, pmax, (pmin, pmax), step=(pmax-pmin)/100)
df = df[(df[price_col] >= sel_price[0]) & (df[price_col] <= sel_price[1])]

# Sqft range
if sqft_living:
    smin, smax = float(df[sqft_living].min()), float(df[sqft_living].max())
    sel_sqft = st.sidebar.slider("Living sqft range", smin, smax, (smin, smax), step=(smax-smin)/100)
    df = df[(df[sqft_living] >= sel_sqft[0]) & (df[sqft_living] <= sel_sqft[1])]

# Bedrooms / Bathrooms
if bed_col:
    beds = sorted(df[bed_col].dropna().unique().tolist())
    sel_beds = st.sidebar.multiselect("Bedrooms", beds, default=beds)
    if sel_beds:
        df = df[df[bed_col].isin(sel_beds)]

if bath_col:
    baths = sorted(df[bath_col].dropna().unique().tolist())
    sel_baths = st.sidebar.multiselect("Bathrooms", baths, default=baths)
    if sel_baths:
        df = df[df[bath_col].isin(sel_baths)]

# Renovated filter
if "renovated_flag" in df.columns:
    only_renov = st.sidebar.checkbox("Show renovated homes only", value=False)
    if only_renov:
        df = df[df["renovated_flag"]]

# Year built filter
if yr_built_col:
    ymin, ymax = int(df[yr_built_col].min()), int(df[yr_built_col].max())
    sel_year = st.sidebar.slider("Year built", ymin, ymax, (ymin, ymax))
    df = df[(df[yr_built_col] >= sel_year[0]) & (df[yr_built_col] <= sel_year[1])]

st.sidebar.markdown("---")
st.sidebar.caption("Upload a different CSV to compare instantly.")

# -------------------------
# Title & Description
# -------------------------
st.title("🏠 USA Housing Prices — Interactive Dashboard")
st.caption("Filter by state/city/price/beds/baths. Upload a CSV to explore another dataset.")

# -------------------------
# KPIs
# -------------------------
left, right = st.columns([1, 3])
left.write(f"**Rows:** {len(df):,} | **Columns:** {df.shape[1]}")

k1, k2, k3, k4 = st.columns(4)
kpi_card(k1, "Median Price", float(df[price_col].median()), prefix="$")
if sqft_living:
    kpi_card(k2, "Median Living Sqft", float(df[sqft_living].median()))
else:
    k2.metric("Median Living Sqft", "—")
kpi_card(k3, "Median Beds", float(df[bed_col].median()) if bed_col else np.nan)
kpi_card(k4, "Median Baths", float(df[bath_col].median()) if bath_col else np.nan)

st.markdown("---")

# -------------------------
# Chart 1: Price vs Living Area (sampled)
# -------------------------
if sqft_living:
    sample_df = df.sample(min(len(df), 6000), random_state=42)
    color_arg = bed_col if bed_col in sample_df.columns else None
    fig1 = px.scatter(
        sample_df,
        x=sqft_living, y=price_col,
        color=color_arg,
        trendline="ols",
        labels={sqft_living: "Living Area (sqft)", price_col: "Price"},
        title="Price vs Living Area (with OLS trendline)"
    )
    fig1.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig1, use_container_width=True)

# -------------------------
# Chart 2: Median Price by City (Top N by count)
# -------------------------
if city_col and city_col in df.columns and len(df[city_col].dropna()) > 0:
    top_n = 15
    top_cities = df[city_col].value_counts().nlargest(top_n).index
    g = df[df[city_col].isin(top_cities)].groupby(city_col, as_index=False)[price_col].median()
    g = g.sort_values(price_col, ascending=False)
    fig2 = px.bar(
        g,
        x=city_col, y=price_col,
        labels={city_col: "City", price_col: "Median Price"},
        title=f"Median Price by City (Top {top_n} by listing count)"
    )
    fig2.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Chart 3: Price per Sqft by City
# -------------------------
if "price_per_sqft" in df.columns and city_col in df.columns:
    g2 = df.dropna(subset=["price_per_sqft"])
    if len(g2) > 0:
        top_n2 = 15
        top_cities2 = g2[city_col].value_counts().nlargest(top_n2).index
        dd = g2[g2[city_col].isin(top_cities2)].groupby(city_col, as_index=False)["price_per_sqft"].median()
        dd = dd.sort_values("price_per_sqft", ascending=False)
        fig3 = px.bar(
            dd, x=city_col, y="price_per_sqft",
            labels={"price_per_sqft": "Median $/sqft"},
            title=f"Median Price per Sqft by City (Top {top_n2})"
        )
        fig3.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# Chart 4: Renovation premium (boxplot)
# -------------------------
if "renovated_flag" in df.columns:
    fig4 = px.box(
        df.dropna(subset=[price_col, "renovated_flag"]),
        x="renovated_flag", y=price_col,
        labels={"renovated_flag": "Renovated", price_col: "Price"},
        title="Renovation Premium (Price Distribution)"
    )
    fig4.update_xaxes(tickvals=[0, 1], ticktext=["No", "Yes"])
    fig4.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig4, use_container_width=True)

# -------------------------
# Optional: Map (if lat/long exist)
# -------------------------
if lat_col and long_col and lat_col in df.columns and long_col in df.columns:
    map_sample = df.dropna(subset=[lat_col, long_col, price_col]).sample(min(3000, len(df)), random_state=1)
    fig_map = px.scatter_mapbox(
        map_sample,
        lat=lat_col, lon=long_col,
        color=price_col,
        size=sqft_living if sqft_living in map_sample.columns else None,
        color_continuous_scale="Viridis",
        zoom=8,
        height=520,
        hover_data=[city_col] if city_col in map_sample.columns else None,
        title="Geographic Distribution of Listings"
    )
    fig_map.update_layout(mapbox_style="carto-positron", margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_map, use_container_width=True)

# -------------------------
# Extras: Price Distribution & Correlation
# -------------------------
with st.expander("More: Price Distribution & Correlation"):
    c1, c2 = st.columns(2)

    with c1:
        fig_hist = px.histogram(
            df, x=price_col, nbins=60,
            title="Price Distribution", labels={price_col: "Price"}
        )
        fig_hist.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        # limit to avoid giant matrices
        if numeric_df.shape[1] > 14:
            numeric_df = numeric_df.iloc[:, :14]
        corr = numeric_df.corr(numeric_only=True)
        fig_corr = px.imshow(
            corr, text_auto=True, aspect="auto",
            title="Correlation Heatmap (numeric columns)"
        )
        fig_corr.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_corr, use_container_width=True)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption(
    "Data visualized with Streamlit • Use the filters on the left to explore cities, price bands, "
    "sqft ranges, and renovated homes. Save static images to include in your policy brief."
)

