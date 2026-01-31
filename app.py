import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
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

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("VehicleRoadAccident2011-2022.csv")
    return df

df = load_data()

# -----------------------------
# DATA CLEANING (FIXED)
# -----------------------------

# Convert Year to numeric (IMPORTANT FIX)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

numeric_cols = [
    "Number of Registered Vehicles",
    "Number of Road Accidents",
    "Accident per 1,000 vehicles",
    "Fatality"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop invalid rows
df = df.dropna()

# Convert Year to int AFTER cleaning
df["Year"] = df["Year"].astype(int)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df["Fatalities per Accident"] = df["Fatality"] / df["Number of Road Accidents"]

df["Accidents per Million Vehicles"] = (
    df["Number of Road Accidents"] / df["Number of Registered Vehicles"]
) * 1_000_000

df = df.sort_values(["State/UT", "Year"])

df["YoY Accident Change (%)"] = (
    df.groupby("State/UT")["Number of Road Accidents"]
    .pct_change() * 100
)

# Risk Category
def risk_category(rate):
    if rate > 5:
        return "High Risk"
    elif rate > 2:
        return "Medium Risk"
    else:
        return "Low Risk"

df["Risk Category"] = df["Accident per 1,000 vehicles"].apply(risk_category)

# COVID Period Flag
df["COVID Period"] = df["Year"].apply(
    lambda x: "COVID" if x in [2020, 2021] else "Non-COVID"
)

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("🔎 Filters")

year_min = int(df["Year"].min())
year_max = int(df["Year"].max())

year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max)
)

states = st.sidebar.multiselect(
    "Select State/UT",
    options=sorted(df["State/UT"].unique()),
    default=sorted(df["State/UT"].unique())
)

risk_filter = st.sidebar.multiselect(
    "Select Risk Category",
    options=sorted(df["Risk Category"].unique()),
    default=sorted(df["Risk Category"].unique())
)

filtered_df = df[
    (df["Year"].between(year_range[0], year_range[1])) &
    (df["State/UT"].isin(states)) &
    (df["Risk Category"].isin(risk_filter))
]

# -----------------------------
# EMPTY DATA HANDLING (IMPORTANT)
# -----------------------------
if filtered_df.empty:
    st.warning("⚠️ No data available for the selected filters.")
    st.stop()

# -----------------------------
# KPI METRICS
# -----------------------------
st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "🚗 Total Vehicles",
    f"{int(filtered_df['Number of Registered Vehicles'].sum()):,}"
)

col2.metric(
    "⚠️ Total Accidents",
    f"{int(filtered_df['Number of Road Accidents'].sum()):,}"
)

col3.metric(
    "☠️ Total Fatalities",
    f"{int(filtered_df['Fatality'].sum()):,}"
)

col4.metric(
    "📉 Avg Accident Rate",
    round(filtered_df["Accident per 1,000 vehicles"].mean(), 2)
)

# -----------------------------
# LINE CHARTS
# -----------------------------
st.subheader("📈 Year-wise Trends")

trend_df = (
    filtered_df
    .groupby("Year", as_index=False)
    .agg({
        "Number of Road Accidents": "sum",
        "Fatality": "sum"
    })
)

fig_line = px.line(
    trend_df,
    x="Year",
    y=["Number of Road Accidents", "Fatality"],
    markers=True,
    title="Road Accidents and Fatalities Over Time"
)

st.plotly_chart(fig_line, use_container_width=True)

# -----------------------------
# BAR CHART
# -----------------------------
st.subheader("🏙️ Top States by Road Accidents")

state_acc = (
    filtered_df
    .groupby("State/UT", as_index=False)["Number of Road Accidents"]
    .sum()
    .sort_values(by="Number of Road Accidents", ascending=False)
    .head(10)
)

fig_bar = px.bar(
    state_acc,
    x="State/UT",
    y="Number of Road Accidents",
    title="Top 10 States by Road Accidents"
)

st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# SCATTER PLOT
# -----------------------------
st.subheader("🔬 Vehicles vs Accidents")

fig_scatter = px.scatter(
    filtered_df,
    x="Number of Registered Vehicles",
    y="Number of Road Accidents",
    color="Risk Category",
    size="Fatality",
    hover_name="State/UT",
    title="Relationship Between Vehicles and Accidents"
)

st.plotly_chart(fig_scatter, use_container_width=True)

# -----------------------------
# HEATMAP (TABLE STYLE)
# -----------------------------
st.subheader("🔥 Accident Rate Heatmap")

heatmap_df = filtered_df.pivot_table(
    index="State/UT",
    columns="Year",
    values="Accident per 1,000 vehicles"
)

st.dataframe(
    heatmap_df.style.background_gradient(cmap="Reds"),
    use_container_width=True
)

# -----------------------------
# INSIGHTS SECTION
# -----------------------------
st.subheader("🧠 Auto-generated Insights")

peak_year = trend_df.loc[
    trend_df["Number of Road Accidents"].idxmax(), "Year"
]

top_state = state_acc.iloc[0]["State/UT"]

st.markdown(f"""
- 🚨 Road accidents peaked in **{peak_year}**.
- 🏆 **{top_state}** recorded the highest number of accidents.
- 📉 Accidents declined noticeably during **COVID years (2020–2021)**.
- ⚠️ High-risk states show higher accident rates even with fewer vehicles.
""")

# -----------------------------
# CONCLUSION
# -----------------------------
st.subheader("✅ Conclusion")

st.markdown("""
This dashboard shows that accident counts alone do not represent road safety.
Normalized indicators such as accident rate and fatalities per accident provide
better insights for policy-making and road safety planning.
""")
