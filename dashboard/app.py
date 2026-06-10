import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

# Resolve workspace paths relatively on the host system execution context
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
TIMESFM_SRC = os.path.join(ROOT_DIR, "timesfm", "src")

if TIMESFM_SRC not in sys.path:
    sys.path.append(TIMESFM_SRC)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import matplotlib.pyplot as plt
import streamlit.components.v1 as components
try:
    import mpld3
    from mpld3 import plugins
except ImportError:
    st.error("Missing dependency: Please run `../timesfm/.venv/bin/pip install mpld3` to enable interactive charts.")

from engine import AdaptiveForecastingEngine

# UI Configuration Bounds
st.set_page_config(page_title="Jewelry Predictive Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main-header { font-size:36px !important; font-weight: bold; color: #1E3A8A; }
    .metric-box { background-color: #F3F4F6; padding: 15px; border-radius: 10px; border-left: 5px solid #3B82F6; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">💎 Jewelry Demand & Predictive Analytics Dashboard</p>', unsafe_allow_html=True)
st.caption("Zero-Shot Multi-Parameter Forecast Pipeline Powered by Google TimesFM 2.5 Transformer Foundation Core")

@st.cache_resource
def get_forecasting_engine():
    engine = AdaptiveForecastingEngine()
    engine.initialize_and_compile()
    return engine

engine = get_forecasting_engine()

# --- SIDEBAR INTERFACE CONTROL SYSTEM ---
st.sidebar.header("📁 Data Sourcing Management")
uploaded_file = st.sidebar.file_uploader("Upload Target Sheet (.xlsx, .csv)", type=["csv", "xlsx"])
# CRITICAL UPDATE: Increased max_value to 52 to allow users to pull a full 1-year horizon forecast
horizon = st.sidebar.slider("Forecast Planning Window (Horizon)", min_value=4, max_value=52, value=12)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".xlsx") or uploaded_file.name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="windows-1252")
        
    # --- AUTO DATA CLEANING: STRIP COMMAS FROM NUMERIC COLUMNS ---
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            # Exclude empty strings and nan-like values when testing for numeric check
            clean_check = df[col].dropna().astype(str).str.strip()
            clean_check = clean_check[~clean_check.str.lower().isin(["nan", "none", "null", ""])]
            
            non_null_series = clean_check.str.replace(r'[,\s]', '', regex=True)
            if len(non_null_series) > 0:
                is_numeric_like = non_null_series.str.match(r'^-?\d+(\.\d+)?$').all()
                if is_numeric_like or pd.to_numeric(non_null_series, errors='coerce').notna().sum() > len(non_null_series) * 0.8:
                    # Clean the entire column
                    clean_series = df[col].astype(str).str.replace(r'[,\s]', '', regex=True)
                    df[col] = pd.to_numeric(clean_series, errors='coerce')

    all_columns = df.columns.tolist()
    
    st.write("### 🔍 Historical Parameter Workspace Preview")
    st.dataframe(df.head(6), use_container_width=True)
    
    st.sidebar.subheader("🛠️ Column Interface Mapping")
    
    # 1. Date Axis can accept any column (string, date, etc.)
    date_column = st.sidebar.selectbox("Select Timestamp/Date Axis", all_columns, index=0)
    
    # 🔥 UX FIX (Option 2): Filter to only show numeric columns for the Target Sales metric
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols:
        st.sidebar.error("Error: No numeric columns found in the uploaded file to select as a target metric.")
        target_column = None
    else:
        # Find a safe default index for Total_Sales if it exists, otherwise default to 0
        default_target_idx = 0
        for idx, col in enumerate(numeric_cols):
            if "sales" in col.lower() or "total" in col.lower():
                default_target_idx = idx
                break
                
        target_column = st.sidebar.selectbox(
            "Select Primary Dependent Target (Sales)", 
            numeric_cols, 
            index=default_target_idx
        )
    
    if target_column:
        # Auto-isolate remaining columns to look for features/parameters
        residual_features = [col for col in all_columns if col not in [date_column, target_column]]
        
        st.sidebar.subheader("📈 Parametric Constraints")
        chosen_features = st.sidebar.multiselect(
            "Select active tracking parameters and festivals to evaluate",
            options=residual_features,
            default=residual_features
        )
        
        if st.sidebar.button("🚀 Run Analytical Prediction Pass"):
            if not chosen_features:
                st.warning("Please specify at least one tracking parameter or festival calendar metric to compute.")
            else:
                with st.spinner("Executing dynamic structural regression matching and transformer tokenized inference locally..."):
                    historical_cleaned_df, future_timeline, forecast_points, forecast_quantiles = engine.execute_modular_forecast(
                        dataframe=df,
                        date_col=date_column,
                        target_col=target_column,
                        selected_features=chosen_features,
                        horizon=horizon
                    )
                    
                    st.write("### 📈 Interactive Strategic Projections Matrix")
                    # --- INTERACTIVE PLOTLY VISUALIZATION ENGINE ---
                    import plotly.graph_objects as go

                    fig = go.Figure()

                    # 1. Interactive Historical Recorded Data Vector Line
                    fig.add_trace(go.Scatter(
                        x=historical_cleaned_df[date_column],
                        y=historical_cleaned_df[target_column],
                        mode='lines+markers',
                        name=f"Historical {target_column}",
                        line=dict(color='#3B82F6', width=2.5),
                        marker=dict(size=4),
                        hovertemplate=f"<b>Date:</b> %{{x}}<br><b>{target_column}:</b> %{{y:,.0f}}<extra></extra>"
                    ))

                    # 2. Interactive Predicted Projection Horizon Vector Line
                    fig.add_trace(go.Scatter(
                        x=future_timeline,
                        y=forecast_points,
                        mode='lines',
                        name=f"TimesFM Projected {target_column}",
                        line=dict(color='#10B981', width=2.5, dash='dash'),
                        hovertemplate=f"<b>Date:</b> %{{x}}<br><b>Projected {target_column}:</b> %{{y:,.0f}}<extra></extra>"
                    ))

                    # 3. Layer the Uncertainty Confidence Ribbon Cushion (Upper & Lower Bounds)
                    fig.add_trace(go.Scatter(
                        x=list(future_timeline) + list(future_timeline)[::-1],
                        y=list(forecast_quantiles[:, 9]) + list(forecast_quantiles[:, 1])[::-1],
                        fill='toself',
                        fillcolor='rgba(16, 185, 129, 0.12)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo='skip',
                        name="80% Statistical Confidence Zone",
                        showlegend=True
                    ))

                    # 4. Premium Dark-Mode Layout & Dynamic Label Styling Pass
                    fig.update_layout(
                        title=dict(
                            text=f"Interactive {target_column} Prediction Analysis Profile",
                            font=dict(size=16, color='#F3F4F6', family="sans-serif"),
                            pad=dict(b=10)
                        ),
                        paper_bgcolor='#0E1117',
                        plot_bgcolor='#1E2430',
                        hovermode='x unified',
                        margin=dict(l=20, r=20, t=50, b=20),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            font=dict(color='#F3F4F6', size=11)
                        ),
                        xaxis=dict(
                            title=dict(text=f"{date_column}", font=dict(color='#9CA3AF', size=12, weight="bold")),
                            tickfont=dict(color='#F3F4F6', size=10),
                            gridcolor='#374151',
                            zerolinecolor='#374151',
                            showgrid=True
                        ),
                        yaxis=dict(
                            title=dict(text=f"{target_column}", font=dict(color='#9CA3AF', size=12, weight="bold")),
                            tickfont=dict(color='#F3F4F6', size=10),
                            gridcolor='#374151',
                            zerolinecolor='#374151',
                            showgrid=True,
                            tickformat=',.0f' # Automatically formats values with commas (e.g., 2,500)
                        )
                    )

                    # Render the responsive chart into the Streamlit dashboard workspace
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("### 🎯 Core Quantitative Operational Indicators")
                    kpi_1, kpi_2, kpi_3 = st.columns(3)
                    
                    with kpi_1:
                        st.markdown(f"""<div class='metric-box'>
                            <p style='color:#4B5563; font-size:14px; margin:0;'>Immediate Horizon Target</p>
                            <p style='color:#111827; font-size:24px; font-weight:bold; margin:5px 0 0 0;'>${forecast_points[0]:,.2f}</p>
                        </div>""", unsafe_allow_html=True)
                        
                    with kpi_2:
                        st.markdown(f"""<div class='metric-box'>
                            <p style='color:#4B5563; font-size:14px; margin:0;'>Peak Capacity Revenue Strain</p>
                            <p style='color:#10B981; font-size:24px; font-weight:bold; margin:5px 0 0 0;'>${np.max(forecast_points):,.2f}</p>
                        </div>""", unsafe_allow_html=True)
                        
                    with kpi_3:
                        st.markdown(f"""<div class='metric-box'>
                            <p style='color:#4B5563; font-size:14px; margin:0;'>Pessimistic Capital Security Floor (P10)</p>
                            <p style='color:#EF4444; font-size:24px; font-weight:bold; margin:5px 0 0 0;'>${np.min(forecast_quantiles[:, 1]):,.2f}</p>
                        </div>""", unsafe_allow_html=True)
                    
                    st.write(" ")
                    st.write("### 📥 Downstream Export Control")
                    export_df = pd.DataFrame({
                        "Date": future_timeline,
                        "Expected_Sales": forecast_points,
                        "P10_Floor_Bound": forecast_quantiles[:, 1],
                        "P90_Ceiling_Bound": forecast_quantiles[:, 9]
                    })
                    
                    st.download_button(
                        label="Download Operational Prediction Metrics Sheet (.csv)",
                        data=export_df.to_csv(index=False),
                        file_name="timesfm_dashboard_export.csv",
                        mime="text/csv"
                    )
else:
    st.info("💡 Open the sidebar, select an Excel sheet containing your store's parameters, and hit compile to map multi-variable demand analysis trends.")