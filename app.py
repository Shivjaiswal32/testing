import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Indian Road Accident Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚦 Indian Road Accident Analysis (2011–2022)")
st.markdown(
    "Interactive dashboard for analyzing road accidents, vehicles, and fatalities "
    "across Indian States and Union Territories."
)

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("VehicleRoadAccident2011-2022.csv")

df = load_data()

# =====================================================
# DATA CLEANING (BULLETPROOF)
# =====================================================
df.columns = df.columns.str.strip()

if "Year" not in df.columns:
    for col in df.columns:
        if col.lower() == "year":
            df.rename(columns={col: "Year"}, inplace=True)

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

numeric_cols = [
    "Number of Registered Vehicles",
    "Number of Road Accidents",
    "Accident per 1,000 vehicles",
    "Fatality"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Year"])
df["Year"] = df["Year"].astype(int)

if df.empty:
    st.error("Dataset is empty after cleaning. Please check the CSV file.")
    st.stop()

# =====================================================
# FEATURE ENGINEERING
# =====================================================
df["Fatalities per Accident"] = df["Fatality"] / df["Number of Road Accidents"]

df["Accidents per Million Vehicles"] = (
    df["Number of Road Accidents"] / df["Number of Registered Vehicles"]
) * 1_000_000

df = df.sort_values(["State/UT", "Year"])

df["YoY Accident Change (%)"] = (
    df.groupby("State/UT")["Number of Road Accidents"]
    .pct_change() * 100
)

def risk_category(rate):
    if rate > 5:
        return "High Risk"
    elif rate > 2:
        return "Medium Risk"
    else:
        return "Low Risk"

df["Risk Category"] = df["Accident per 1,000 vehicles"].apply(risk_category)

df["COVID Period"] = df["Year"].apply(
    lambda x: "COVID" if x in [2020, 2021] else "Non-COVID"
)

# =====================================================
# SIDEBAR FILTERS
# =====================================================
st.sidebar.header("🔎 Filters")

year_min = int(df["Year"].min())
year_max = int(df["Year"].max())

year_range = st.sidebar.slider(
    "Select Year Range",
    year_min,
    year_max,
    (year_min, year_max)
)

states = st.sidebar.multiselect(
    "Select State/UT",
    sorted(df["State/UT"].unique()),
    sorted(df["State/UT"].unique())
)

risk_filter = st.sidebar.multiselect(
    "Select Risk Category",
    sorted(df["Risk Category"].unique()),
    sorted(df["Risk Category"].unique())
)

filtered_df = df[
    (df["Year"].between(year_range[0], year_range[1])) &
    (df["State/UT"].isin(states)) &
    (df["Risk Category"].isin(risk_filter))
]

if filtered_df.empty:
    st.warning("⚠️ No data available for selected filters.")
    st.stop()

filtered_df = filtered_df.copy()
filtered_df["Fatality"] = filtered_df["Fatality"].fillna(0).clip(lower=0)

# =====================================================
# KPI METRICS
# =====================================================
st.subheader("📊 Key Performance Indicators")

c1, c2, c3, c4 = st.columns(4)

c1.metric("🚗 Total Vehicles", f"{int(filtered_df['Number of Registered Vehicles'].sum()):,}")
c2.metric("⚠️ Total Accidents", f"{int(filtered_df['Number of Road Accidents'].sum()):,}")
c3.metric("☠️ Total Fatalities", f"{int(filtered_df['Fatality'].sum()):,}")
c4.metric("📉 Avg Accident Rate", round(filtered_df["Accident per 1,000 vehicles"].mean(), 2))

# =====================================================
# LINE CHART
# =====================================================
st.subheader("📈 Year-wise Trends")

trend_df = filtered_df.groupby("Year", as_index=False).agg({
    "Number of Road Accidents": "sum",
    "Fatality": "sum"
})

fig_line = px.line(
    trend_df,
    x="Year",
    y=["Number of Road Accidents", "Fatality"],
    markers=True,
    title="Road Accidents and Fatalities Over Time"
)

st.plotly_chart(fig_line, use_container_width=True)

# =====================================================
# BAR CHART
# ==========================
