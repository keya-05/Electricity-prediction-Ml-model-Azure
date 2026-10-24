"""Live dashboard for the energy edge predictions.

Usage:
  streamlit run dashboard.py
"""
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import altair as alt
import pandas as pd
import requests
import streamlit as st

from config import BASE_DIR, DB_PATH, EDGE_URL
from household import HOMES
from storage import clear_predictions, connect

ACTUAL_COLOR, PREDICTED_COLOR = "#2a78d6", "#eb6834"
MUTED = "#898781"
LEVEL_ICONS = {"Low": "🟢", "Normal": "🔵", "High": "🟠", "Peak": "🔴"}

st.set_page_config(page_title="Smart Energy Edge AI", page_icon="⚡", layout="wide")


def is_unusual(actual, predicted):
    # well above what the model expected for that hour, e.g. an appliance left on
    return actual - predicted > max(0.8, 0.5 * predicted)


def load_predictions(limit):
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", conn, params=(limit,))
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["received_at"] = pd.to_datetime(df["received_at"], utc=True, format="ISO8601")
    df["home"] = df["device_id"].map(lambda d: f"{d} · {HOMES[d]['label']}" if d in HOMES else d)
    return df.sort_values("id")


def hourly_history(df):
    """One row per home per completed hour: actual meter kWh vs the model's prediction for that hour.

    A reading at hour t carries prev_kwh = actual use in hour t-1, and the reading at t-1
    carried the prediction for hour t-1.
    """
    df = df.sort_values(["device_id", "timestamp"]).copy()
    df["hour_start"] = df["timestamp"] - timedelta(hours=1)
    df["actual_kwh"] = df["prev_kwh"]
    df["expected_kwh"] = df.groupby("device_id")["predicted_kwh"].shift(1)
    hist = df.dropna(subset=["expected_kwh"])
    hist = hist[hist["hour_start"] == hist.groupby("device_id")["timestamp"].shift(1).reindex(hist.index)]
    hist = hist.assign(unusual=[is_unusual(a, p) for a, p in zip(hist["actual_kwh"], hist["expected_kwh"])])
    return hist[["device_id", "home", "hour_start", "actual_kwh", "expected_kwh", "unusual"]]


def edge_online():
    try:
        return requests.get(f"{EDGE_URL}/health", timeout=1).ok
    except requests.RequestException:
        return False


def actual_vs_predicted_chart(hist, forecast, home):
    rows = pd.concat([
        hist.assign(series="Actual", kwh=hist["actual_kwh"]),
        hist.assign(series="Predicted", kwh=hist["expected_kwh"]),
        forecast.assign(series="Predicted", kwh=forecast["predicted_kwh"], hour_start=forecast["timestamp"]),
    ])[["hour_start", "series", "kwh"]]
    hover = alt.selection_point(fields=["hour_start"], nearest=True, on="pointerover", clear="pointerout", empty=False)
    base = alt.Chart(rows).encode(
        x=alt.X("hour_start:T", title=None, axis=alt.Axis(format="%a %H:%M", labelColor=MUTED, grid=False, tickCount=4)),
        y=alt.Y("kwh:Q", title="kWh per hour", scale=alt.Scale(zero=True)),
        color=alt.Color("series:N", scale=alt.Scale(domain=["Actual", "Predicted"], range=[ACTUAL_COLOR, PREDICTED_COLOR]),
                        legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=["Actual", "Predicted"], range=[[1, 0], [5, 3]]), legend=None),
    )
    lines = base.mark_line(strokeWidth=2)
    points = base.mark_circle(size=70, stroke="white", strokeWidth=2).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("hour_start:T", title="Hour", format="%a %H:%M"),
                 alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("kwh:Q", title="kWh", format=".2f")],
    ).add_params(hover)
    rule = alt.Chart(rows).mark_rule(color=MUTED, strokeWidth=1).encode(x="hour_start:T").transform_filter(hover)
    return alt.layer(lines, rule, points).properties(title=home, height=230)


def load_profile_chart(hist):
    profile = (hist.assign(hour=hist["hour_start"].dt.hour)
               .groupby("hour", as_index=False)["actual_kwh"].mean())
    return alt.Chart(profile).mark_bar(color=ACTUAL_COLOR, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("hour:O", title="Hour of day"),
        y=alt.Y("actual_kwh:Q", title="Average kWh per home"),
        tooltip=[alt.Tooltip("hour:O", title="Hour"), alt.Tooltip("actual_kwh:Q", title="Avg kWh", format=".2f")],
    ).properties(height=220)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    refresh = st.slider("Refresh every (seconds)", 1, 10, 2)
    hours = st.select_slider("Simulated hours to show", options=[24, 48, 72, 168], value=48)
    metrics_file = BASE_DIR / "model_metrics.json"
    if metrics_file.exists():
        m = json.loads(metrics_file.read_text())
        st.header("Model")
        st.markdown(
            f"**{m['model']}**  \n"
            f"Test MAE: **{m['mae']:.3f} kWh** (R² {m['r2']:.2f})  \n"
            f"Naive baseline MAE: {m['baseline_mae']:.3f} kWh  \n"
            f"Trained on {m['train_rows']:,} hourly rows"
        )
        st.caption("Feature importance")
        st.dataframe(
            pd.DataFrame(m["feature_importance"].items(), columns=["Feature", "Importance"])
            .sort_values("Importance", ascending=False),
            hide_index=True, width="stretch",
        )
    st.divider()
    if st.button("Clear prediction history"):
        clear_predictions(connect())
        st.toast("History cleared")

st.title("⚡ Smart Energy Edge AI – Real-time Consumption Prediction")
st.caption("Simulated IoT smart meters → Edge module (Random Forest) → Next-hour kWh forecast per home  ·  1 simulated hour every 2 s")


@st.fragment(run_every=refresh)
def live_view():
    df = load_predictions(hours * max(len(HOMES), 1))
    online = edge_online()

    status = st.columns(5)
    status[0].metric("Edge module", "🟢 Online" if online else "🔴 Offline")
    if df.empty:
        st.info("No predictions yet. Start the edge module and the simulator (see README).")
        return
    hist = hourly_history(df)
    age = (datetime.now(timezone.utc) - df["received_at"].max()).total_seconds()
    status[1].metric("Last reading", f"{age:.0f}s ago")
    status[2].metric("Predictor", df["predictor"].iloc[-1].capitalize())
    status[3].metric("Avg inference time", f"{df['inference_ms'].mean():.1f} ms")
    if not hist.empty:
        normal = hist[~hist["unusual"]]
        live_mae = (normal["actual_kwh"] - normal["expected_kwh"]).abs().mean()
        status[4].metric("Live MAE", f"{live_mae:.2f} kWh", help="Average |actual − predicted| over completed hours, excluding unusual-usage hours")

    # Per-home cards
    st.subheader("Next-hour forecast")
    device_ids = [d for d in HOMES if d in set(df["device_id"])]
    cols = st.columns(len(device_ids))
    for col, device_id in zip(cols, device_ids):
        now = df[df["device_id"] == device_id].iloc[-1]
        last = hist[hist["device_id"] == device_id].tail(1)
        with col.container(border=True):
            st.markdown(f"#### {device_id}")
            st.caption(f"{HOMES[device_id]['label']} · next hour starts {now['timestamp']:%a %H:%M}")
            st.markdown(f"### {now['predicted_kwh']:.2f} kWh  \n{LEVEL_ICONS[now['load_level']]} {now['load_level']} load expected")
            a, b = st.columns(2)
            a.metric("Last hour actual (kWh)", f"{now['prev_kwh']:.2f}")
            if not last.empty:
                b.metric("Last hour predicted (kWh)", f"{last['expected_kwh'].iloc[0]:.2f}")
            a.metric("Temperature (°C)", f"{now['temperature']:.1f}")
            b.metric("People home", f"{now['occupancy']}")
            if not last.empty and last["unusual"].iloc[0]:
                st.warning(f"⚠️ Unusual usage: {last['actual_kwh'].iloc[0]:.2f} kWh used vs "
                           f"{last['expected_kwh'].iloc[0]:.2f} kWh expected. Check for an appliance left on.")

    # Actual vs predicted, one small multiple per home on a shared scale
    st.subheader("Actual vs predicted consumption")
    chart_cols = st.columns(len(device_ids))
    for c, device_id in zip(chart_cols, device_ids):
        h = hist[hist["device_id"] == device_id]
        f = df[df["device_id"] == device_id].tail(1)
        c.altair_chart(actual_vs_predicted_chart(h, f, device_id), width="stretch", theme="streamlit")
    st.caption("Solid = metered consumption; dashed = model prediction (the last dashed point is the forecast for the coming hour).")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Daily load profile")
        if not hist.empty:
            st.altair_chart(load_profile_chart(hist), width="stretch", theme="streamlit")
    with right:
        st.subheader("Unusual-usage alerts")
        alerts = hist[hist["unusual"]].sort_values("hour_start", ascending=False)
        if alerts.empty:
            st.success("No unusual usage in this window.")
        else:
            st.dataframe(
                alerts.assign(extra_kwh=alerts["actual_kwh"] - alerts["expected_kwh"])[
                    ["hour_start", "device_id", "actual_kwh", "expected_kwh", "extra_kwh"]
                ].round(2),
                hide_index=True, width="stretch", height=220,
            )

    st.subheader("Prediction log")
    log = df.sort_values("id", ascending=False).head(25)[
        ["timestamp", "device_id", "temperature", "humidity", "occupancy", "prev_kwh",
         "predicted_kwh", "load_level", "predictor", "source"]
    ]
    st.dataframe(log, hide_index=True, width="stretch", height=300)


live_view()
