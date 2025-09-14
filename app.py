import io
import math
import datetime as dt
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

st.set_page_config(page_title="CFO Forecasting Workbench", layout="wide")
st.title("📈 CFO Forecasting Workbench")

# -------------------------
# Helpers
# -------------------------
def read_any_file(upload) -> pd.DataFrame:
    if upload is None:
        return pd.DataFrame()
    name = upload.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(upload)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(upload)
    else:
        st.error("Please upload a CSV or Excel file.")
        return pd.DataFrame()

def coerce_datetime(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce", utc=False)
    return s

def infer_frequency(index: pd.DatetimeIndex) -> Optional[str]:
    try:
        f = pd.infer_freq(index)
        return f
    except Exception:
        return None

def train_test_split_series(y: pd.Series, horizon: int) -> Tuple[pd.Series, pd.Series]:
    """Last `horizon` points reserved for backtest."""
    if horizon >= len(y):
        return y, pd.Series(dtype=y.dtype)
    return y.iloc[:-horizon], y.iloc[-horizon:]

def add_time_features(dt_index: pd.DatetimeIndex, add_month_dummies: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(index=dt_index)
    df["t"] = np.arange(1, len(dt_index) + 1)
    df["year"] = dt_index.year
    df["month"] = dt_index.month
    df["quarter"] = dt_index.quarter
    df["dow"] = dt_index.dayofweek
    if add_month_dummies:
        # capture seasonality for regression/ML
        m = pd.get_dummies(df["month"].astype("category"), prefix="m", drop_first=True)
        df = pd.concat([df, m], axis=1)
    return df

def ensure_monotonic(ts: pd.Series) -> pd.Series:
    ts = ts.sort_index()
    ts = ts[~ts.index.duplicated(keep="first")]
    return ts

def rmse(y_true, y_pred) -> float:
    a = np.array(y_true)
    b = np.array(y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) == len(b) and len(a) > 0 else float("nan")

def export_bytes(df: pd.DataFrame) -> Tuple[bytes, bytes]:
    csv_bytes = df.to_csv(index=True).encode("utf-8")
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Forecast")
    xlsx_bytes = out.getvalue()
    return csv_bytes, xlsx_bytes

# -------------------------
# Sidebar: Data & Setup
# -------------------------
with st.sidebar:
    st.header("1) Data")
    upl = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"])
    df = read_any_file(upl)
    if not df.empty:
        st.write("Preview:", df.head(5))

    st.header("2) Core Settings")
    date_col = st.selectbox("Date/Period column", options=(df.columns.tolist() if not df.empty else []))
    target_col = st.selectbox("Target (to forecast)", options=(df.columns.tolist() if not df.empty else []))
    exog_cols = st.multiselect("Optional drivers (exogenous)", options=(df.columns.tolist() if not df.empty else []))
    horizon = st.number_input("Forecast horizon (# periods)", min_value=1, max_value=120, value=12, step=1)
    backtest = st.checkbox("Perform simple backtest (last horizon as test)", value=True)

    st.header("3) Method")
    method = st.selectbox(
        "Choose forecasting method",
        options=[
            "Trend-Based (Straight Line)",
            "Moving Average",
            "Exponential Smoothing",
            "Seasonal Decomposition (STL)",
            "Time Series (Holt-Winters)",
            "Regression (with optional drivers)",
            "AI-Based (Random Forest)",
            "Expert Judgment Overlay",
        ],
    )

    st.caption("Methods aligned with your decision matrix.")

# Guard clauses
if df.empty or not date_col or not target_col:
    st.info("Upload data and select the date and target columns to begin.")
    st.stop()

# -------------------------
# Prepare Series
# -------------------------
data = df.copy()
data[date_col] = coerce_datetime(data[date_col])
data = data.dropna(subset=[date_col, target_col])
data = data.sort_values(date_col).set_index(date_col)

# aggregate duplicates by mean (common in exports)
y = data[target_col].astype("float").groupby(level=0).mean().copy()
y = ensure_monotonic(y)

# try to set frequency
freq = infer_frequency(y.index)
if freq is None:
    st.warning("Could not infer frequency. Set one below (default monthly).")
    freq = st.selectbox("Frequency", ["D", "W", "M", "Q", "Y"], index=2)
else:
    st.success(f"Inferred frequency: {freq}")
y = y.asfreq(freq)

# Optional exogenous
X = None
if exog_cols:
    X = data[exog_cols].groupby(level=0).mean().reindex(y.index).interpolate().copy()

# Backtest split
y_train, y_test = train_test_split_series(y, horizon) if backtest else (y, pd.Series(dtype=float))
if X is not None:
    X_train = X.loc[y_train.index]
    X_test = X.loc[y_test.index] if not y_test.empty else None

# -------------------------
# Modeling Functions
# -------------------------
def forecast_trend_line(y_train: pd.Series, horizon: int) -> pd.Series:
    Xf = add_time_features(y_train.index, add_month_dummies=False)[["t"]]
    lr = LinearRegression().fit(Xf, y_train.values)
    # future index
    future_index = pd.date_range(start=y_train.index[-1] + pd.tseries.frequencies.to_offset(freq),
                                 periods=horizon, freq=freq)
    X_future = add_time_features(future_index, add_month_dummies=False)[["t"]]
    preds = lr.predict(X_future)
    return pd.Series(preds, index=future_index, name="forecast")

def forecast_moving_average(y_train: pd.Series, horizon: int, window: int = 3) -> pd.Series:
    avg = y_train.rolling(window=window, min_periods=1).mean().iloc[-1]
    # naive extension: repeat last moving average
    future_index = pd.date_range(start=y_train.index[-1] + pd.tseries.frequencies.to_offset(freq),
                                 periods=horizon, freq=freq)
    return pd.Series([avg]*horizon, index=future_index, name="forecast")

def forecast_exp_smoothing(y_train: pd.Series, horizon: int, seasonal: Optional[str] = None, sp: Optional[int] = None):
    if seasonal:
        model = ExponentialSmoothing(
            y_train, trend="add", seasonal=seasonal, seasonal_periods=sp, initialization_method="estimated"
        ).fit(optimized=True)
    else:
        model = SimpleExpSmoothing(y_train, initialization_method="estimated").fit(optimized=True)
    f = model.forecast(horizon)
    f.name = "forecast"
    return f

def decompose_stl(y_full: pd.Series, sp: int = 12):
    stl = STL(y_full, period=sp, robust=True)
    res = stl.fit()
    return res

def forecast_holt_winters(y_train: pd.Series, horizon: int, seasonal: str = "add", sp: int = 12):
    model = ExponentialSmoothing(
        y_train, trend="add", seasonal=seasonal, seasonal_periods=sp, initialization_method="estimated"
    ).fit(optimized=True)
    f = model.forecast(horizon)
    f.name = "forecast"
    return f

def forecast_regression(y_train: pd.Series, horizon: int, X_train: Optional[pd.DataFrame], X_test_future: Optional[pd.DataFrame]):
    # Build features (time & seasonal dummies) + exogenous if provided
    F_train = add_time_features(y_train.index)
    if X_train is not None:
        F_train = F_train.join(X_train)
    lr = LinearRegression().fit(F_train, y_train.values)

    future_index = pd.date_range(start=y_train.index[-1] + pd.tseries.frequencies.to_offset(freq),
                                 periods=horizon, freq=freq)
    F_future = add_time_features(future_index)
    if X_test_future is not None:
        F_future = F_future.join(X_test_future.reindex(future_index))  # may be NaN if not supplied
        F_future = F_future.fillna(method="ffill").fillna(method="bfill")

    preds = lr.predict(F_future)
    return pd.Series(preds, index=future_index, name="forecast")

def forecast_random_forest(y_train: pd.Series, horizon: int, X_train: Optional[pd.DataFrame], X_full: Optional[pd.DataFrame]):
    """Direct multi-step with expanding features & seasonal dummies."""
    F_train = add_time_features(y_train.index)
    if X_train is not None:
        F_train = F_train.join(X_train)
    rf = RandomForestRegressor(n_estimators=400, random_state=42)
    rf.fit(F_train, y_train.values)

    # Recursive strategy: roll forward using only time + (optional) X if provided
    future_index = pd.date_range(start=y_train.index[-1] + pd.tseries.frequencies.to_offset(freq),
                                 periods=horizon, freq=freq)
    F_future = add_time_features(future_index)
    if X_full is not None:
        # If user didn't supply future exog, use last available values
        last_row = X_full.iloc[[-1]].copy()
        X_future = pd.concat([last_row]*horizon, ignore_index=True)
        X_future.index = future_index
        F_future = F_future.join(X_future)
    preds = rf.predict(F_future)
    return pd.Series(preds, index=future_index, name="forecast")

# -------------------------
# Controls per method
# -------------------------
col1, col2, col3 = st.columns(3)
with col1:
    seasonal_periods = st.number_input("Seasonal period (e.g., 12 for monthly data)", 2, 366, value=12, step=1)
with col2:
    ma_window = st.number_input("Moving Average window", 1, 60, value=3, step=1)
with col3:
    expert_adj = st.number_input("Expert judgment overlay (% add on forecast)", -100.0, 200.0, value=0.0, step=1.0)

# -------------------------
# Fit / Forecast
# -------------------------
st.subheader("Results")
fig_area = st.empty()

# Backtest fit -> produce in-sample forecast for test window (simple holdout)
bt_rmse = None
if method == "Trend-Based (Straight Line)":
    f = forecast_trend_line(y_train, horizon)
    if backtest and not y_test.empty:
        # naive approach: refit without last horizon, compare to actual test timestamps
        bt_pred = forecast_trend_line(y_train, len(y_test)).reindex(y_test.index)
        bt_rmse = rmse(y_test, bt_pred)

elif method == "Moving Average":
    f = forecast_moving_average(y_train, horizon, window=ma_window)
    if backtest and not y_test.empty:
        bt_pred = forecast_moving_average(y_train, len(y_test), window=ma_window).reindex(y_test.index)
        bt_rmse = rmse(y_test, bt_pred)

elif method == "Exponential Smoothing":
    use_seasonal = st.checkbox("Use seasonal (Holt-Winters) smoothing", value=True)
    if use_seasonal:
        f = forecast_exp_smoothing(y_train, horizon, seasonal="add", sp=seasonal_periods)
    else:
        f = forecast_exp_smoothing(y_train, horizon, seasonal=None, sp=None)
    if backtest and not y_test.empty:
        bt_pred = forecast_exp_smoothing(y_train, len(y_test), seasonal=("add" if use_seasonal else None), sp=seasonal_periods).reindex(y_test.index)
        bt_rmse = rmse(y_test, bt_pred)

elif method == "Seasonal Decomposition (STL)":
    res = decompose_stl(y, sp=seasonal_periods)
    st.write("**STL Components**")
    st.line_chart(pd.DataFrame({"Observed": y, "Trend": res.trend, "Seasonal": res.seasonal, "Remainder": res.resid}))
    st.caption("Use insight from decomposition to pick a forecasting method. Below we fit Holt-Winters on the trend+seasonal pattern.")
    f = forecast_holt_winters(y_train, horizon, seasonal="add", sp=seasonal_periods)
    if backtest and not y_test.empty:
        bt_pred = forecast_holt_winters(y_train, len(y_test), seasonal="add", sp=seasonal_periods).reindex(y_test.index)
        bt_rmse = rmse(y_test, bt_pred)

elif method == "Time Series (Holt-Winters)":
    seasonal_type = st.selectbox("Seasonal type", ["add", "mul"], index=0)
    f = forecast_holt_winters(y_train, horizon, seasonal=seasonal_type, sp=seasonal_periods)
    if backtest and not y_test.empty:
        bt_pred = forecast_holt_winters(y_train, len(y_test), seasonal=seasonal_type, sp=seasonal_periods).reindex(y_test.index)
        bt_rmse = rmse(y_test, bt_pred)

elif method == "Regression (with optional drivers)":
    if X is None or X.empty:
        st.info("No drivers selected—using time/seasonal features only.")
        X_future = None
    else:
        # Future drivers not provided—hold last value constant (documented)
        X_future = X
    f = forecast_regression(y_train, horizon, (X_train if exog_cols else None), (X_future if exog_cols else None))
    if backtest and not y_test.empty:
        bt_pred = forecast_regression(y_train, len(y_test), (X_train if exog_cols else None), (X_future if exog_cols else None)).reindex(y_test.index)
        bt_rmse = rmse(y_test, bt_pred)

elif method == "AI-Based (Random Forest)":
    f = forecast_random_forest(y_train, horizon, (X_train if exog_cols else None), (X if exog_cols else None))
    if backtest and not y_test.empty:
        bt_pred = forecast_random_forest(y_train, len(y_test), (X_train if exog_cols else None), (X if exog_cols else None)).reindex(y_test.index)
        bt_rmse = rmse(y_test, bt_pred)

elif method == "Expert Judgment Overlay":
    # Base: Holt-Winters, then overlay expert %
    base = forecast_holt_winters(y_train, horizon, seasonal="add", sp=seasonal_periods)
    f = base.copy()
else:
    st.stop()

# Expert overlay (applies to all if slider != 0)
if expert_adj != 0.0:
    f = f * (1.0 + expert_adj / 100.0)

# -------------------------
# Assemble Output
# -------------------------
hist_df = pd.DataFrame({"actual": y})
fc_df = pd.DataFrame({"forecast": f})
out_df = pd.concat([hist_df, fc_df], axis=0)

# Charts
st.write("**History & Forecast**")
st.line_chart(out_df)

# Metrics
if bt_rmse is not None and not math.isnan(bt_rmse):
    st.metric("Backtest RMSE (last horizon)", f"{bt_rmse:,.2f}")

# Table
st.write("**Forecast table**")
st.dataframe(fc_df)

# Downloads
csv_bytes, xlsx_bytes = export_bytes(fc_df)
colA, colB, colC = st.columns([1,1,2])
with colA:
    st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="forecast.csv", mime="text/csv")
with colB:
    st.download_button("⬇️ Download Excel", data=xlsx_bytes, file_name="forecast.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Guidance
with st.expander("ℹ️ Method Guidance (CFO Matrix)"):
    st.markdown("""
- **Short-Term Cash (0–3m):** Exponential Smoothing / Moving Average  
- **Working Capital / CCC:** Holt–Winters + STL (seasonality)  
- **Budget & Quarterly Reforecast:** Regression (+ Expert Judgment)  
- **Long-Term Strategy (1–5y):** Regression / AI-Based (with drivers)  
- **Scenario & Sensitivity:** Regression / AI with driver tweaks  
- **Volatile Environments:** Exponential Smoothing + Expert Overlay  
- **Seasonal Businesses:** STL + Holt–Winters  
- **Regulatory / ESG:** AI + Expert Judgment (drivers often external)
""")
