"""Live dashboard for the weather edge predictions.

Usage:
  streamlit run dashboard.py
"""
import json
import sqlite3
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import requests
import streamlit as st

from config import BASE_DIR, DB_PATH, EDGE_URL
from storage import clear_predictions, connect

STATION_COLORS = {"ws-pune": "#2a78d6", "ws-mumbai": "#eb6834", "ws-nashik": "#1baf7a"}
STATION_NAMES = {"ws-pune": "Pune", "ws-mumbai": "Mumbai", "ws-nashik": "Nashik"}
SENSORS = [
    ("temperature", "Temperature", "°C"),
    ("humidity", "Humidity", "%"),
    ("pressure", "Pressure", "hPa"),
    ("wind_speed", "Wind speed", "km/h"),
]
MUTED = "#898781"

st.set_page_config(page_title="Weather Edge AI", page_icon="🌦️", layout="wide")


def load_predictions(limit):
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", conn, params=(limit,)
        )
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601").dt.tz_convert(None)
    df["station"] = df["device_id"].map(STATION_NAMES).fillna(df["device_id"])
    return df.sort_values("id")


def edge_online():
    try:
        return requests.get(f"{EDGE_URL}/health", timeout=1).ok
    except requests.RequestException:
        return False


def station_scale(df):
    ids = [d for d in STATION_COLORS if d in set(df["device_id"])]
    ids += sorted(set(df["device_id"]) - set(ids))
    names = [STATION_NAMES.get(d, d) for d in ids]
    colors = [STATION_COLORS.get(d, MUTED) for d in ids]
    return alt.Scale(domain=names, range=colors)


def line_chart(df, field, title, unit, height=220, threshold=None, y_domain=None):
    hover = alt.selection_point(
        fields=["timestamp"], nearest=True, on="pointerover", clear="pointerout", empty=False
    )
    y = alt.Y(f"{field}:Q", title=unit, scale=alt.Scale(zero=False, domain=y_domain) if y_domain else alt.Scale(zero=False))
    base = alt.Chart(df).encode(
        x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%H:%M:%S", labelColor=MUTED, grid=False)),
        y=y,
        color=alt.Color("station:N", scale=station_scale(df), legend=alt.Legend(orient="top", title=None)),
    )
    lines = base.mark_line(strokeWidth=2)
    points = base.mark_circle(size=80, stroke="white", strokeWidth=2).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip("station:N", title="Station"),
            alt.Tooltip("timestamp:T", title="Time", format="%H:%M:%S"),
            alt.Tooltip(f"{field}:Q", title=f"{title} ({unit})", format=".2f" if field == "rain_trend" else ".1f"),
            alt.Tooltip("prediction:N", title="Prediction"),
        ],
    ).add_params(hover)
    rule = alt.Chart(df).mark_rule(color=MUTED, strokeWidth=1).encode(x="timestamp:T").transform_filter(hover)
    layers = [lines, rule, points]
    if threshold is not None:
        layers.insert(0, alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(
            color=MUTED, strokeDash=[4, 4], strokeWidth=1
        ).encode(y="y:Q"))
    return alt.layer(*layers).properties(title=title, height=height)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    refresh = st.slider("Refresh every (seconds)", 1, 10, 2)
    window = st.select_slider("Readings to show", options=[60, 150, 300, 600, 1500], value=150)
    metrics_file = BASE_DIR / "model_metrics.json"
    if metrics_file.exists():
        m = json.loads(metrics_file.read_text())
        st.header("Model")
        st.markdown(
            f"**{m['model']}**  \n"
            f"Test accuracy: **{m['accuracy']:.0%}**  \n"
            f"Trained on {m['train_rows']} rows, tested on {m['test_rows']}"
        )
        st.caption("Feature importance")
        st.dataframe(
            pd.DataFrame(m["feature_importance"].items(), columns=["Feature", "Importance"]),
            hide_index=True, width="stretch",
        )
    st.divider()
    if st.button("Clear prediction history"):
        clear_predictions(connect())
        st.toast("History cleared")

st.title("🌦️ Weather Edge AI – Real-time Rain Prediction")
st.caption("Simulated IoT weather stations → Edge module (Decision Tree) → Live predictions")


@st.fragment(run_every=refresh)
def live_view():
    df = load_predictions(window)
    online = edge_online()

    status_cols = st.columns(4)
    status_cols[0].metric("Edge module", "🟢 Online" if online else "🔴 Offline")
    if df.empty:
        st.info("No predictions yet. Start the edge module and the simulator (see README).")
        return
    last_seen = df["timestamp"].max()
    age = (datetime.now(timezone.utc).replace(tzinfo=None) - last_seen).total_seconds()
    status_cols[1].metric("Last reading", f"{age:.0f}s ago")
    status_cols[2].metric("Predictions shown", f"{len(df)}")
    status_cols[3].metric("Avg inference time", f"{df['inference_ms'].mean():.2f} ms")

    # Latest reading per station
    st.subheader("Current conditions")
    device_ids = [d for d in STATION_COLORS if d in set(df["device_id"])]
    device_ids += sorted(set(df["device_id"]) - set(device_ids))
    cols = st.columns(len(device_ids))
    for col, device_id in zip(cols, device_ids):
        rows = df[df["device_id"] == device_id].tail(2)
        now = rows.iloc[-1]
        prev = rows.iloc[0] if len(rows) > 1 else now
        with col.container(border=True):
            rain = now["prediction"] == "Rain"
            st.markdown(f"#### {now['station']}")
            st.markdown(
                f"### {'🌧️ Rain' if rain else '☀️ No Rain'}  \n"
                f"Rain probability: **{now['rain_probability']:.0%}**"
            )
            a, b = st.columns(2)
            # deltas are neutral (grey): a rise in pressure or humidity is not "good" or "bad"
            a.metric("Temperature (°C)", f"{now['temperature']:.1f}", f"{now['temperature'] - prev['temperature']:+.1f}", delta_color="off")
            b.metric("Humidity (%)", f"{now['humidity']:.0f}", f"{now['humidity'] - prev['humidity']:+.0f}", delta_color="off")
            a.metric("Pressure (hPa)", f"{now['pressure']:.0f}", f"{now['pressure'] - prev['pressure']:+.1f}", delta_color="off")
            b.metric("Wind (km/h)", f"{now['wind_speed']:.1f}", f"{now['wind_speed'] - prev['wind_speed']:+.1f}", delta_color="off")

    # Rain probability over time
    # A decision tree's probability jumps between leaves, so plot a short rolling mean
    st.subheader("Rain probability")
    df["rain_trend"] = df.groupby("device_id")["rain_probability"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    st.altair_chart(
        line_chart(df, "rain_trend", "Rain probability, 5-reading rolling average (dashed line = 0.5 threshold)",
                   "probability", height=240, threshold=0.5, y_domain=[0, 1]),
        width="stretch", theme="streamlit",
    )

    # Sensor telemetry, one chart per measure
    st.subheader("Sensor telemetry")
    for row in (SENSORS[:2], SENSORS[2:]):
        c1, c2 = st.columns(2)
        for c, (field, title, unit) in zip((c1, c2), row):
            c.altair_chart(line_chart(df, field, title, unit), width="stretch", theme="streamlit")

    # Summary + log
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Rain alerts")
        summary = (
            df.assign(rain=df["prediction"].eq("Rain"))
            .groupby("station")
            .agg(Readings=("rain", "size"), Rain=("rain", "sum"))
            .assign(**{"Rain %": lambda t: (100 * t["Rain"] / t["Readings"]).round(0)})
            .reset_index()
            .rename(columns={"station": "Station"})
        )
        st.dataframe(summary, hide_index=True, width="stretch")
    with right:
        st.subheader("Prediction log")
        log = df.sort_values("id", ascending=False).head(25)[
            ["timestamp", "station", "temperature", "humidity", "pressure", "wind_speed",
             "prediction", "rain_probability", "source"]
        ]
        st.dataframe(log, hide_index=True, width="stretch", height=300)


live_view()
