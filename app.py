"""Streamlit frontend for the AI Capacity Forecaster Agent."""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from agent.controller import AgentController
from charts.plotter import create_forecast_chart, create_multi_metric_chart
from utils.constants import ForecastModel, METRIC_DISPLAY_NAMES, QueryIntent
from utils.csv_loader import CSVLoadError, load_csv, load_default_sample

st.set_page_config(
    page_title="AI Capacity Forecaster",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Session State ───────────────────────────────────────────────────────────
def init_state():
    for key, default in [
        ("df", None),
        ("results", None),
        ("data_source", "sample"),  # "sample" or "upload"
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


# ─── Sidebar ─────────────────────────────────────────────────────────────────
def sidebar():
    st.sidebar.title("⚙️ Configuration")

    # Data Source Selection (radio to avoid checkbox toggle issues)
    data_source = st.sidebar.radio(
        "Data Source",
        ["Sample Dataset", "Upload CSV"],
        index=0 if st.session_state.data_source == "sample" else 1,
    )

    if data_source == "Upload CSV":
        uploaded = st.sidebar.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Required columns: date, cpu_usage, memory_usage, disk_usage",
        )
        if uploaded is not None:
            try:
                df = load_csv(uploaded)
                # Validate date column is properly parsed
                if df["date"].dtype != "datetime64[ns]":
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    if df["date"].isna().any():
                        st.sidebar.error("❌ Date column could not be parsed. Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format.")
                        return model
                st.session_state.df = df
                st.session_state.data_source = "upload"
                st.sidebar.success(f"✅ Loaded {len(df)} rows spanning {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load CSV: {e}")
                st.sidebar.info("📋 Expected columns: date, cpu_usage, memory_usage, disk_usage")
        elif st.session_state.data_source == "upload":
            st.sidebar.info("Upload a CSV file to begin.")
    else:
        if st.session_state.data_source != "sample" or st.session_state.df is None:
            try:
                st.session_state.df = load_default_sample()
                st.session_state.data_source = "sample"
                st.sidebar.success("✅ Sample data loaded")
            except CSVLoadError as e:
                st.sidebar.error(f"❌ {e}")

    # Model Selection
    st.sidebar.subheader("Forecast Model")
    model_choice = st.sidebar.radio(
        "Model", ["Linear Regression", "ARIMA"],
        index=0,
        label_visibility="collapsed",
    )
    model = ForecastModel.ARIMA if model_choice == "ARIMA" else ForecastModel.LINEAR

    # API Status
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key and api_key != "your_gemini_api_key_here":
        st.sidebar.success("🤖 Gemini AI: Connected")
    else:
        st.sidebar.info("🔧 Add GEMINI_API_KEY to .env for AI NL parsing")

    st.sidebar.divider()

    # Quick Queries
    st.sidebar.subheader("🔍 Quick Queries")
    sample_queries = [
        "When will disk usage hit 80%?",
        "Forecast CPU usage for next 6 months.",
        "Which resource will exceed threshold first?",
        "Show risk analysis for all resources.",
        "What if CPU grows 20% faster?",
        "Forecast memory usage for next 3 months.",
    ]
    for q in sample_queries:
        if st.sidebar.button(q, use_container_width=True, type="tertiary"):
            st.session_state.quick_query = q

    # Controls
    if st.session_state.results:
        st.sidebar.divider()
        if st.sidebar.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.results = None

    return model


# ─── Main UI ─────────────────────────────────────────────────────────────────
def main():
    init_state()
    model = sidebar()
    df = st.session_state.df

    st.title("📊 AI Capacity Forecaster")
    st.markdown(
        "AI-powered infrastructure capacity planning with "
        "natural language queries and predictive forecasting."
    )

    if df is None:
        st.info("📤 **No data loaded.** Select a data source from the sidebar.")
        return

    # Data Summary
    st.caption(
        f"📅 {df['date'].iloc[0]} → {df['date'].iloc[-1]} | "
        f"{len(df)} rows | "
        f"Source: {st.session_state.data_source}"
    )

    # Live Dashboard
    st.subheader("📊 Current Utilization")
    latest = df.iloc[-1]
    cols = st.columns(3)
    for i, col in enumerate(["cpu_usage", "memory_usage", "disk_usage"]):
        v = latest[col]
        delta = f"{v - df.iloc[-2][col]:+.1f}%" if len(df) > 1 else None
        color = "normal" if v < 60 else "inverse" if v < 80 else "off"
        with cols[i]:
            st.metric(
                METRIC_DISPLAY_NAMES.get(col, col),
                f"{v:.1f}%",
                delta,
                delta_color=color,
            )

    with st.expander("📋 Raw Data Preview"):
        st.dataframe(df.tail(10), width=800)

    st.divider()

    # ─── Query Section ────────────────────────────────────────────────────
    st.subheader("💬 Ask a Question")

    # Handle quick query from sidebar
    default_query = st.session_state.get("quick_query", "")
    if default_query:
        del st.session_state["quick_query"]

    query = st.text_input(
        "Type your question below and click Analyze:",
        value=default_query,
        placeholder="e.g., When will disk usage hit 80%?",
        label_visibility="collapsed",
        key="query_input",
    )

    if default_query:
        # Auto-run when a quick query button was clicked
        run_analysis = True
    else:
        run_analysis = st.button("🔮 Analyze", type="primary", use_container_width=True)

    if run_analysis:
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("🤖 Analyzing your query..."):
                try:
                    ctrl = AgentController(model=model)
                    resp = ctrl.run(query.strip(), df)
                    st.session_state.results = (query.strip(), resp)
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    st.info(
                        "💡 Make sure your CSV has columns: "
                        "date, cpu_usage, memory_usage, disk_usage"
                    )

    # ─── Results ──────────────────────────────────────────────────────────
    if st.session_state.results:
        _show_results(st.session_state.results)


def _show_results(results):
    query, resp = results
    st.divider()
    st.subheader(f"📈 Results for: _{query}_")

    # Metrics
    c1, c2, c3 = st.columns(3)
    conf = resp.confidence
    icon = "🟢" if conf.overall >= 0.7 else "🟡" if conf.overall >= 0.4 else "🔴"
    with c1:
        st.metric("Confidence", f"{icon} {conf.overall*100:.0f}%", conf.label)

    risk = resp.risk_report.overall_risk.value
    risk_map = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
    with c2:
        st.metric("Risk Level", f"{risk_map.get(risk, '')} {risk}")

    intent = resp.parsed.intent.value.replace("_", " ").title()
    intent_map = {
        "Threshold Crossing": "⏰", "Forecast Period": "📈",
        "Compare Resources": "🏆", "Risk Analysis": "⚠️",
        "What If": "🔮", "General": "💬",
    }
    with c3:
        st.metric("Intent", f"{intent_map.get(intent, '📊')} {intent}")

    # Response
    st.markdown("### 📋 Analysis")
    st.markdown(resp.answer)

    # Recommendations
    if resp.recommendations:
        with st.expander("💡 Recommendations", expanded=True):
            for rec in resp.recommendations:
                c = "🔴"
                for k, v in {"P0": "🔴", "P1": "🟠", "P2": "🟡", "P3": "🟢"}.items():
                    if k in rec.priority:
                        c = v
                        break
                st.markdown(f"**{c} {rec.display_name}** — {rec.risk_level.value}")
                st.markdown(f"> {rec.action}")
                if rec.rationale:
                    st.caption(f"📝 {rec.rationale}")
                st.markdown("---")

    # Chart
    st.markdown("### 📈 Forecast Chart")
    try:
        p = resp.parsed
        if p.intent in (QueryIntent.RISK_ANALYSIS, QueryIntent.COMPARE_RESOURCES):
            fig = create_multi_metric_chart(resp.forecasts)
        else:
            fig = create_forecast_chart(resp.forecasts[p.metrics[0]])
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Chart could not be generated for this query type.")

    # Technical Details
    with st.expander("🔍 Technical Details"):
        for metric, fc in resp.forecasts.items():
            name = METRIC_DISPLAY_NAMES.get(metric, metric)
            end = fc.forecast_values[-1] if fc.forecast_values else "N/A"
            cross = (
                fc.threshold_crossing_date.strftime("%Y-%m-%d")
                if fc.threshold_crossing_date
                else "Not within horizon"
            )
            st.markdown(
                f"**{name}**  \n"
                f"- Model: `{fc.model_type.upper()}` | "
                f"R²: `{fc.r_squared:.3f}` | Slope: `{fc.slope:.3f}%/mo`  \n"
                f"- Current: `{fc.historical_values[-1]:.1f}%` → "
                f"Projected: `{end}%`  \n"
                f"- Threshold crossing: `{cross}`  \n"
                f"---"
            )


if __name__ == "__main__":
    main()