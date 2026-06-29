import streamlit as st
import pandas as pd
import numpy as np

PAGE_TITLE   = "Experienced Energy - Enterprise Monitor"
PAGE_ICON    = "🔋"
SPIKE_MULTIPLIER = 1.4
CO2_PER_KWH      = 0.79
GST_RATE         = 0.10

REQUIRED_COLS = [
    "reading_date",
    "location",
    "kwh_consumed",
    "costing_type",
    "total_incl_gst_aud",
    "devices_left_on_after_hours",
    "spike_detected",
    "usage_flag",
    "co2_kg_emitted",
    "efficiency_score",
    "peak_demand_kw",
    "pct_vs_location_avg"
]

GRANULARITY_MAP = {
    "Daily":     "D",
    "Weekly":    "W",
    "Monthly":   "M",
    "Quarterly": "Q"
}

def validate_csv(df):
    missing = []
    for col in REQUIRED_COLS:
        if col not in df.columns:
            missing.append(col)
    return missing


def load_and_prepare(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)

        missing = validate_csv(df)
        if missing:
            st.error(f"❌ CSV is missing required columns: {', '.join(missing)}")
            st.info("Please upload the correct energy_data_complete.csv or energy_sample.csv file.")
            return None

        df["reading_date"] = pd.to_datetime(df["reading_date"])

        df["month_period"]   = df["reading_date"].dt.to_period("M")
        df["week_period"]    = df["reading_date"].dt.to_period("W")
        df["quarter_period"] = df["reading_date"].dt.to_period("Q")

        numeric_cols = [
            "kwh_consumed", "total_incl_gst_aud",
            "devices_left_on_after_hours", "co2_kg_emitted",
            "efficiency_score", "peak_demand_kw", "pct_vs_location_avg"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        return None


def apply_filters(df, date_range, location, costing_type, day_type):
    filtered = df.copy()

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = date_range
        filtered = filtered[
            (filtered["reading_date"].dt.date >= start_d) &
            (filtered["reading_date"].dt.date <= end_d)
        ]

    if location != "All Locations":
        filtered = filtered[filtered["location"] == location]

    if costing_type == "Peak only":
        filtered = filtered[filtered["costing_type"] == "peak"]
    elif costing_type == "Off-peak only":
        filtered = filtered[filtered["costing_type"] == "off_peak"]

    if day_type == "Weekdays only":
        filtered = filtered[filtered["day_type"] == "weekday"]
    elif day_type == "Weekends only":
        filtered = filtered[filtered["day_type"] == "weekend"]

    return filtered


def aggregate_by_granularity(df, granularity):
    period_code = GRANULARITY_MAP.get(granularity, "D")

    df = df.copy()
    df["period"] = df["reading_date"].dt.to_period(period_code).astype(str)

    agg = df.groupby("period").agg(
        kwh          =("kwh_consumed",                "sum"),
        cost         =("total_incl_gst_aud",          "sum"),
        devices_on   =("devices_left_on_after_hours", "sum"),
        co2          =("co2_kg_emitted",               "sum"),
        peak_demand  =("peak_demand_kw",               "max"),
        avg_eff      =("efficiency_score",             "mean"),
    ).reset_index()

    for col in ["kwh", "cost", "co2", "avg_eff"]:
        agg[col] = agg[col].round(2)

    return agg


def calculate_kpis(df):
    kpis = {}

    kpis["total_kwh"]      = round(df["kwh_consumed"].sum(), 1)
    kpis["total_cost"]     = round(df["total_incl_gst_aud"].sum(), 2)
    kpis["avg_daily_kwh"]  = round(
        df.groupby("reading_date")["kwh_consumed"].sum().mean(), 1
    )
    kpis["peak_demand_kw"] = round(df["peak_demand_kw"].max(), 1)
    kpis["total_co2"]      = round(df["co2_kg_emitted"].sum(), 1)
    kpis["devices_left_on"]= int(df["devices_left_on_after_hours"].sum())
    kpis["avg_efficiency"] = round(df["efficiency_score"].mean(), 0)
    kpis["spike_count"]    = int((df["spike_detected"] == "YES").sum())
    kpis["alert_count"]    = int((df["usage_flag"] == "ALERT_HIGH").sum())

    kpis["action_required"] = (kpis["spike_count"] > 0) or (kpis["alert_count"] > 0)

    return kpis


def get_worst_alert(df, flag_col, flag_val):
    subset = df[df[flag_col] == flag_val]
    if subset.empty:
        return None
    return subset.nlargest(1, "kwh_consumed").iloc[0]


def get_quarterly_summary(df):
    df = df.copy()
    df["qtr"] = df["reading_date"].dt.to_period("Q").astype(str)

    agg_dict = {
        "total_kwh":       ("kwh_consumed",                "sum"),
        "total_bill":      ("total_incl_gst_aud",          "sum"),
        "spike_days":      ("spike_detected",              lambda x: (x == "YES").sum()),
        "devices_left_on": ("devices_left_on_after_hours", "sum"),
        "total_co2":       ("co2_kg_emitted",               "sum"),
    }

    optional = {
        "usage_cost":    "usage_cost_aud",
        "supply_charge": "supply_charge_aud",
        "gst":           "gst_aud",
    }
    for label, col in optional.items():
        if col in df.columns:
            agg_dict[label] = (col, "sum")

    summary = df.groupby("qtr").agg(**agg_dict).reset_index()

    rename_map = {
        "qtr":           "Quarter",
        "total_kwh":     "Total kWh",
        "total_bill":    "Total Bill ($)",
        "spike_days":    "Spike Days",
        "devices_left_on": "Devices Left On",
        "total_co2":     "CO₂ (kg)",
        "usage_cost":    "Usage Cost ($)",
        "supply_charge": "Supply Charge ($)",
        "gst":           "GST ($)",
    }
    summary = summary.rename(columns={k: v for k, v in rename_map.items() if k in summary.columns})

    for col in summary.select_dtypes(include=[np.number]).columns:
        summary[col] = summary[col].round(2)

    return summary


st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="collapsedControl"] { display: none !important; }
    section[data-testid="stSidebar"] { min-width: 260px !important; width: 260px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .stApp { background-color: #F8F8F6; }

    [data-testid="stSidebar"] {
        background-color: #26215C;
    }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stDateInput label {
        color: #CCCCFF !important;
        font-size: 0.85rem !important;
    }

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 12px;
        padding: 16px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    [data-testid="stMetricLabel"] { font-size: 0.82rem; color: #5F5E5A; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; color: #26215C; font-weight: 700; }

    .alert-box {
        background: #FEE2E2;
        border-left: 5px solid #DC2626;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 12px 0;
    }
    .alert-box h4 { color: #991B1B; margin: 0 0 6px 0; font-size: 1rem; }
    .alert-box p  { color: #7F1D1D; margin: 0; font-size: 0.88rem; }

    .section-header {
        font-size: 1rem;
        font-weight: 700;
        color: #26215C;
        border-bottom: 2px solid #534AB7;
        padding-bottom: 4px;
        margin: 20px 0 12px 0;
    }
    .dash-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #26215C;
        letter-spacing: -0.5px;
    }
    .dash-subtitle {
        font-size: 0.9rem;
        color: #5F5E5A;
        margin-top: -6px;
        margin-bottom: 10px;
    }

    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


date_range        = None
granularity       = "Daily"
selected_loc      = "All Locations"
costing_type_opt  = "All"
day_type_opt      = "All"


with st.sidebar:
    st.markdown("## 🔋 Experienced Energy")
    st.markdown("**Enterprise Energy Monitor**")
    st.markdown("---")
    st.markdown("### 📂 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload energy CSV",
        type=["csv"],
        help="Upload energy_data_complete.csv or energy_sample.csv"
    )


if uploaded_file is not None:
    df_result = load_and_prepare(uploaded_file)
    if df_result is not None:
        st.session_state.df = df_result
else:
    if "df" not in st.session_state:
        st.session_state.df = None

df = st.session_state.get("df", None)


with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔍 Filters")

    if df is not None:

        all_months      = sorted(df["reading_date"].dt.to_period("M").unique())
        month_labels    = [m.strftime("%b %Y") for m in all_months]
        month_start_map = {m.strftime("%b %Y"): m.start_time.date() for m in all_months}
        month_end_map   = {m.strftime("%b %Y"): m.end_time.date()   for m in all_months}

        start_month = st.selectbox(
            "📅  Start month",
            month_labels,
            index=0
        )
        end_month = st.selectbox(
            "📅  End month",
            month_labels,
            index=len(month_labels) - 1
        )

        start_date = month_start_map[start_month]
        end_date   = month_end_map[end_month]

        if end_date < start_date:
            start_date, end_date = end_date, start_date

        date_range = (start_date, end_date)

        granularity = st.selectbox(
            "📊  Time Period",
            list(GRANULARITY_MAP.keys())
        )

        loc_options = ["All Locations"] + sorted(
            df["location"].dropna().unique().tolist()
        )
        selected_loc = st.selectbox("🏢  Location", loc_options)

        costing_type_opt = st.radio(
            "⚡  Rate Type",
            ["All", "Peak only", "Off-peak only"]
        )

        day_type_opt = st.radio(
            "📆  Day",
            ["All", "Weekdays only", "Weekends only"]
        )

    st.markdown("---")


if df is None:
    st.markdown('<div class="dash-title">🔋 Experienced Energy</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dash-subtitle">Enterprise Energy Monitoring Dashboard</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**📂 Step 1 — Upload**\n\nUpload `energy_sample.csv` using the sidebar.")
    with c2:
        st.info("**🔍 Step 2 — Filter**\n\nFilter by date range, location, costing type and granularity.")
    with c3:
        st.info("**📊 Step 3 — Analyse**\n\nView KPIs, usage trends, alerts and environmental impact.")
    st.markdown("---")
    st.markdown(
        "**About this system:** Experienced Energy transforms raw electricity consumption CSV data "
        "into clear, actionable dashboards for enterprise managers, facilities coordinators "
        "and administrators."
    )

if df is not None:

    filtered = apply_filters(df, date_range, selected_loc, costing_type_opt, day_type_opt)

    if filtered.empty:
        st.warning("⚠️ No data matches your current filters. Try adjusting the date range or location.")

    else:

        agg_df = aggregate_by_granularity(filtered, granularity)

        kpis = calculate_kpis(filtered)

        title_loc = selected_loc if selected_loc != "All Locations" else "All Locations"
        date_from = filtered["reading_date"].min().strftime("%d %b %Y")
        date_to   = filtered["reading_date"].max().strftime("%d %b %Y")

        st.markdown(f'<div class="dash-title">🔋 {title_loc}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="dash-subtitle">'
            f'{granularity} view  |  {date_from} → {date_to}  |  '
            f'{len(filtered):,} readings loaded'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")

        if kpis["action_required"]:
            st.markdown(
                '<div class="section-header">🚨 Action Required</div>',
                unsafe_allow_html=True
            )
            a1, a2 = st.columns(2)

            with a1:
                worst_alert = get_worst_alert(filtered, "usage_flag", "ALERT_HIGH")
                if worst_alert is not None:
                    pct = worst_alert.get("pct_vs_location_avg", 0)
                    st.markdown(f"""
                    <div class="alert-box">
                        <h4>⚠️ High Usage Detected</h4>
                        <p><strong>{worst_alert['location']}</strong> recorded
                        <strong>{worst_alert['kwh_consumed']:.1f} kWh</strong> on
                        <strong>{pd.to_datetime(worst_alert['reading_date']).strftime('%d %b %Y')}</strong>
                        &nbsp;({float(pct):+.1f}% vs location average)</p>
                    </div>
                    """, unsafe_allow_html=True)

            with a2:
                worst_spike = get_worst_alert(filtered, "spike_detected", "YES")
                if worst_spike is not None:
                    reason = str(worst_spike.get("spike_reason", "unknown")).replace("_", " ")
                    st.markdown(f"""
                    <div class="alert-box">
                        <h4>⚡ Spike Detected</h4>
                        <p><strong>{worst_spike['location']}</strong> — spike recorded on
                        <strong>{pd.to_datetime(worst_spike['reading_date']).strftime('%d %b %Y')}</strong>.
                        Likely cause: <strong>{reason}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown(
            '<div class="section-header">📊 Key Performance Indicators</div>',
            unsafe_allow_html=True
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("⚡ Total kWh Consumed",     f"{kpis['total_kwh']:,.1f} kWh")
        k2.metric("💰 Total Cost (incl. GST)", f"${kpis['total_cost']:,.2f}")
        k3.metric("📊 Avg Daily Usage",         f"{kpis['avg_daily_kwh']:,.1f} kWh/day")
        k4.metric("🔺 Peak Demand Recorded",    f"{kpis['peak_demand_kw']:.1f} kW")

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("🌿 Total CO₂ Emitted",       f"{kpis['total_co2']:,.1f} kg")
        k6.metric("🔌 Devices Left On",          f"{kpis['devices_left_on']:,}")
        k7.metric("🎯 Avg Efficiency Score",     f"{int(kpis['avg_efficiency'])} / 100")
        k8.metric("⚠️ Spike Events",             f"{kpis['spike_count']}")

        st.markdown(
            '<div class="section-header">📈 Usage Trends</div>',
            unsafe_allow_html=True
        )

        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown(f"**Energy Consumption — {granularity}**")
            st.line_chart(
                agg_df.set_index("period")["kwh"],
                use_container_width=True,
                height=280
            )

        with ch2:
            st.markdown(f"**Total Cost — {granularity} ($ incl. GST)**")
            st.bar_chart(
                agg_df.set_index("period")["cost"],
                use_container_width=True,
                height=280
            )

        st.markdown(
            '<div class="section-header">🏢 Usage by Location</div>',
            unsafe_allow_html=True
        )

        lc1, lc2 = st.columns(2)

        with lc1:
            st.markdown("**Total kWh by Location**")
            loc_kwh = (
                filtered.groupby("location")["kwh_consumed"]
                .sum()
                .sort_values(ascending=False)
            )
            st.bar_chart(loc_kwh, use_container_width=True, height=260)

        with lc2:
            st.markdown("**Total Cost by Location ($ AUD)**")
            loc_cost = (
                filtered.groupby("location")["total_incl_gst_aud"]
                .sum()
                .sort_values(ascending=False)
            )
            st.bar_chart(loc_cost, use_container_width=True, height=260)

        st.markdown(
            '<div class="section-header">🔌 Devices Left On After Hours</div>',
            unsafe_allow_html=True
        )

        d1, d2 = st.columns([2, 1])

        with d1:
            st.markdown(f"**Devices Left On — {granularity}**")
            st.bar_chart(
                agg_df.set_index("period")["devices_on"],
                use_container_width=True,
                height=240
            )

        with d2:
            st.markdown("**Total by Location**")
            dev_by_loc = (
                filtered.groupby("location")["devices_left_on_after_hours"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={
                    "location":                      "Location",
                    "devices_left_on_after_hours":   "Devices Left On"
                })
            )
            st.dataframe(dev_by_loc, hide_index=True, use_container_width=True, height=240)

        st.markdown(
            '<div class="section-header">🧾 Quarterly Bill Comparison</div>',
            unsafe_allow_html=True
        )

        quarterly_df = get_quarterly_summary(filtered)
        st.dataframe(quarterly_df, hide_index=True, use_container_width=True)

        st.markdown(
            '<div class="section-header">🌿 Environmental Impact & Efficiency</div>',
            unsafe_allow_html=True
        )

        e1, e2 = st.columns(2)

        with e1:
            st.markdown("**CO₂ Emissions Over Time (kg)**")
            st.area_chart(
                agg_df.set_index("period")["co2"],
                use_container_width=True,
                height=240
            )

        with e2:
            st.markdown("**Average Efficiency Score by Location**")
            eff_by_loc = (
                filtered.groupby("location")["efficiency_score"]
                .mean()
                .sort_values()
            )
            st.bar_chart(eff_by_loc, use_container_width=True, height=240)

        st.markdown(
            '<div class="section-header">🕐 After Hours Usage</div>',
            unsafe_allow_html=True
        )

        if "operating_status" in filtered.columns:
            os1, os2 = st.columns(2)
            with os1:
                status_counts = (
                    filtered["operating_status"]
                    .value_counts()
                    .reset_index()
                    .rename(columns={"operating_status": "Status", "count": "Readings"})
                )
                status_counts["Status"] = status_counts["Status"].str.replace("_", " ").str.title()
                st.dataframe(status_counts, hide_index=True, use_container_width=True)

            with os2:
                outside = filtered[filtered["operating_status"] == "outside_operating_hours"]
                outside_kwh  = round(outside["kwh_consumed"].sum(), 1)
                outside_cost = round(outside["total_incl_gst_aud"].sum(), 2)
                outside_pct  = round((outside_kwh / kpis["total_kwh"]) * 100, 1) if kpis["total_kwh"] > 0 else 0

                st.metric("kWh Used Outside Hours", f"{outside_kwh:,.1f} kWh")
                st.metric("Cost Outside Hours",      f"${outside_cost:,.2f}")
                st.metric("% of Total Usage",        f"{outside_pct}%")

        with st.expander("🗂️ View Raw Data Table"):
            display_cols = [
                "reading_date", "location", "kwh_consumed", "costing_type",
                "total_incl_gst_aud", "devices_left_on_after_hours",
                "spike_detected", "usage_flag", "efficiency_score", "co2_kg_emitted"
            ]
            display_cols = [c for c in display_cols if c in filtered.columns]

            st.dataframe(
                filtered[display_cols].sort_values("reading_date", ascending=False),
                use_container_width=True,
                hide_index=True
            )
            st.caption(f"Showing {len(filtered):,} records. Use filters in the sidebar to narrow results.")

        st.markdown("---")
        st.markdown(
            "<p style='text-align:center;color:#888;font-size:0.8rem;'>"
            "🔋 Experienced Energy — Enterprise Energy Monitoring System"
            "</p>",
            unsafe_allow_html=True
        )
