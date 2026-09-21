import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import kagglehub
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None


# =========================================================
# PAGE
# =========================================================
st.set_page_config(
    page_title="FleetOps AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --bg: #f6f8fb;
    --card: #ffffff;
    --ink: #172033;
    --muted: #6b7280;
    --line: #e7ebf2;
    --accent: #246bfd;
    --accent2: #6c5ce7;
    --good: #0f9d72;
    --warn: #d99000;
    --danger: #c73a3a;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #f8fbff 0%, #f5f7fb 100%);
}
[data-testid="stSidebar"] {
    background: #0f172a;
}
[data-testid="stSidebar"] * {
    color: #f8fafc;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}
.hero {
    background: linear-gradient(110deg, #111a2f 0%, #1b2b53 62%, #274c87 100%);
    border-radius: 22px;
    padding: 1.7rem 1.9rem;
    color: white;
    margin-bottom: 1rem;
    box-shadow: 0 10px 35px rgba(15, 23, 42, .14);
}
.hero-kicker {
    font-size: .82rem;
    letter-spacing: .13em;
    text-transform: uppercase;
    opacity: .75;
    font-weight: 700;
}
.hero-title {
    font-size: 2.3rem;
    font-weight: 850;
    margin: .25rem 0 .25rem 0;
}
.hero-sub {
    font-size: 1.02rem;
    opacity: .83;
    max-width: 850px;
}
.kpi-card {
    background: white;
    border: 1px solid #e8edf4;
    border-radius: 16px;
    padding: 1rem 1rem .85rem 1rem;
    box-shadow: 0 4px 16px rgba(16, 24, 40, .05);
    min-height: 112px;
}
.kpi-label {
    color: #6b7280;
    font-size: .84rem;
    margin-bottom: .35rem;
}
.kpi-value {
    color: #172033;
    font-weight: 800;
    font-size: 1.62rem;
    line-height: 1.1;
}
.kpi-note {
    color: #8a94a6;
    font-size: .78rem;
    margin-top: .45rem;
}
.panel {
    background: white;
    border: 1px solid #e8edf4;
    border-radius: 18px;
    padding: 1rem 1.1rem;
    box-shadow: 0 4px 16px rgba(16, 24, 40, .04);
}
.agent-badge {
    display:inline-block;
    padding:.3rem .65rem;
    border-radius:999px;
    background:#eaf1ff;
    color:#245bd6;
    font-size:.78rem;
    font-weight:700;
}
.best-card {
    background: linear-gradient(145deg, #ffffff 0%, #f7fbff 100%);
    border: 1px solid #dfe9fb;
    border-radius: 18px;
    padding: 1.2rem 1.25rem;
    box-shadow: 0 5px 20px rgba(36,107,253,.08);
}
.best-title {
    font-size:1.5rem;
    font-weight:800;
    color:#172033;
}
.score-pill {
    display:inline-block;
    padding:.34rem .72rem;
    border-radius:999px;
    background:#e8f8f1;
    color:#137555;
    font-weight:800;
}
.muted {
    color:#6b7280;
}
.small {
    font-size:.86rem;
}
div[data-testid="stChatMessage"] {
    border: 1px solid #e8edf4;
    border-radius: 16px;
    padding: .25rem .35rem;
    background: white;
}
.stTabs [data-baseweb="tab-list"] {
    gap: .35rem;
}
.stTabs [data-baseweb="tab"] {
    height: 46px;
    border-radius: 12px 12px 0 0;
    padding-left: 1rem;
    padding-right: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="hero-kicker">AI + Operations • Construction Fleet Intelligence</div>
  <div class="hero-title">FleetOps AI</div>
  <div class="hero-sub">
    An intelligent operations agent that converts project requirements into
    ranked fleet recommendations, explains the decision, and surfaces fleet risks.
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# CONSTANTS
# =========================================================
VEHICLE_TYPES = [
    "Heavy Truck",
    "Electric Truck",
    "Medium Truck",
    "Refrigerated Truck",
    "Light Truck",
]
ROAD_TYPES = ["Mountainous", "Smooth", "Rough", "Moderate", "Congested"]
WEATHER_TYPES = ["Foggy", "Cold", "Humid", "Clear", "Rainy", "Hot"]

SELECTED_COLUMNS = [
    "Vehicle_ID", "Make_and_Model", "Vehicle_Type", "Year_of_Manufacture",
    "Weather_Conditions", "Road_Conditions", "Usage_Hours", "Load_Capacity",
    "Actual_Load", "Engine_Temperature", "Tire_Pressure", "Fuel_Consumption",
    "Battery_Status", "Vibration_Levels", "Oil_Quality", "Brake_Condition",
    "Failure_History", "Anomalies_Detected", "Diagnostic_Trouble_Code_Count",
    "Maintenance_Cost", "Historical_Maintenance_Cost", "Downtime_Maintenance",
    "Impact_on_Efficiency", "Days_Since_Last_Maintenance", "Maintenance_Type",
    "Maintenance_Severity", "Maintenance_Required", "Predictive_Score",
]

RESULT_COLUMNS = [
    "Vehicle_ID", "Make_and_Model", "Vehicle_Type", "Load_Capacity",
    "Fuel_Consumption", "Maintenance_Required", "Maintenance_Severity",
    "Failure_History", "Anomalies_Detected", "Downtime_Maintenance",
    "Road_Conditions", "Weather_Conditions", "Match_Score", "Classification",
]


# =========================================================
# DATA
# =========================================================
@st.cache_data(show_spinner=False)
def read_csv_path(path_str):
    return pd.read_csv(path_str)

@st.cache_data(show_spinner=False)
def read_csv_upload(file):
    return pd.read_csv(file)

def find_dataset():
    filename = "logistics_predictive_maintenanceV2.csv"

    here = Path(__file__).resolve().parent

    # First: check if the dataset already exists locally
    candidates = [
        here / "data" / filename,
        here.parent / "data" / filename,
        Path.cwd() / "data" / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    # Second: automatically download the public dataset from Kaggle
    try:
        downloaded_path = kagglehub.dataset_download(
            "datasetengineer/logistics-vehicle-maintenance-history-dataset",
            path=filename
        )

        downloaded_path = Path(downloaded_path)

        if downloaded_path.exists():
            return downloaded_path

    except Exception as e:
        st.warning(f"Automatic dataset download failed: {e}")

    return None

def prepare_data(df):
    missing = [c for c in SELECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    out = df[SELECTED_COLUMNS].copy()
    numeric = [
        "Usage_Hours", "Load_Capacity", "Actual_Load", "Engine_Temperature",
        "Tire_Pressure", "Fuel_Consumption", "Battery_Status", "Vibration_Levels",
        "Oil_Quality", "Failure_History", "Anomalies_Detected",
        "Diagnostic_Trouble_Code_Count", "Maintenance_Cost",
        "Historical_Maintenance_Cost", "Downtime_Maintenance",
        "Impact_on_Efficiency", "Days_Since_Last_Maintenance",
        "Maintenance_Required", "Predictive_Score",
    ]
    for c in numeric:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=[
        "Vehicle_ID", "Vehicle_Type", "Load_Capacity", "Fuel_Consumption",
        "Road_Conditions", "Weather_Conditions"
    ]).copy()

    out = out[
        (out["Load_Capacity"] >= 0)
        & (out["Fuel_Consumption"] >= 0)
        & (out["Downtime_Maintenance"] >= 0)
    ].copy()
    return out


# =========================================================
# MATCHING ENGINE
# =========================================================
def match_project(data, vehicle_type, min_load, road, weather, top_n=10):
    c = data[
        (data["Vehicle_Type"] == vehicle_type)
        & (data["Load_Capacity"] >= min_load)
    ].copy()

    if c.empty:
        return c, 0

    eligible_count = len(c)

    c["Load_Score"] = (min_load / c["Load_Capacity"]).clip(0, 1)

    sev_map = {
        "Normal": 1.0,
        "Minor Maintenance": 0.60,
        "Major Maintenance": 0.20,
    }
    c["Maintenance_Score"] = c["Maintenance_Severity"].map(sev_map).fillna(.5)
    c.loc[c["Maintenance_Required"] == 0, "Maintenance_Score"] = 1.0

    c["Fuel_Score"] = (
        1 - c["Fuel_Consumption"].rank(method="average", pct=True)
    ).clip(0, 1)

    fail = (1 - c["Failure_History"].rank(method="average", pct=True)).clip(0, 1)
    anom = (1 - c["Anomalies_Detected"].rank(method="average", pct=True)).clip(0, 1)
    down = (1 - c["Downtime_Maintenance"].rank(method="average", pct=True)).clip(0, 1)
    c["Reliability_Score"] = (fail + anom + down) / 3

    c["Road_Score"] = np.where(c["Road_Conditions"] == road, 1.0, 0.0)
    c["Weather_Score"] = np.where(c["Weather_Conditions"] == weather, 1.0, 0.0)

    c["Match_Score"] = (
        c["Load_Score"] * 25
        + c["Maintenance_Score"] * 25
        + c["Fuel_Score"] * 15
        + c["Reliability_Score"] * 15
        + c["Road_Score"] * 10
        + c["Weather_Score"] * 10
    )

    def classify(x):
        if x >= 85: return "Best Match"
        if x >= 70: return "Suitable"
        if x >= 50: return "Attention Required"
        return "Not Recommended"

    c["Classification"] = c["Match_Score"].apply(classify)
    c = c.sort_values(["Match_Score", "Fuel_Consumption"], ascending=[False, True])
    return c[RESULT_COLUMNS].head(top_n).copy(), eligible_count

def add_reason(matches, req):
    if matches.empty:
        return matches
    fuel_med = matches["Fuel_Consumption"].median()

    def reason(r):
        parts = []
        if r["Load_Capacity"] >= req["min_load_capacity"]:
            parts.append(f"meets the load requirement ({r['Load_Capacity']:,.0f})")
        if r["Maintenance_Required"] == 0:
            parts.append("needs no immediate maintenance")
        if r["Failure_History"] == 0:
            parts.append("has no recorded failures")
        if r["Anomalies_Detected"] == 0:
            parts.append("has no detected anomalies")
        if r["Road_Conditions"] == req["road_condition"]:
            parts.append(f"matches {req['road_condition'].lower()} road conditions")
        if r["Weather_Conditions"] == req["weather_condition"]:
            parts.append(f"matches {req['weather_condition'].lower()} weather")
        if r["Fuel_Consumption"] <= fuel_med:
            parts.append("has relatively efficient fuel consumption")
        return "Recommended because it " + ", ".join(parts) + "."

    out = matches.copy()
    out["Why_This_Match"] = out.apply(reason, axis=1)
    out["Match_Score"] = out["Match_Score"].round(1)
    return out.reset_index(drop=True)


# =========================================================
# AI / AGENT
# =========================================================
def gemini_client():
    if genai is None:
        return None
    try:
        key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None
    if not key:
        return None
    return genai.Client(api_key=str(key).strip())

def local_project_parser(text):
    low = text.lower()

    # Vehicle type
    if "heavy truck" in low:
        vehicle_type = "Heavy Truck"
    elif "electric truck" in low:
        vehicle_type = "Electric Truck"
    elif "refrigerated truck" in low:
        vehicle_type = "Refrigerated Truck"
    elif "light truck" in low:
        vehicle_type = "Light Truck"
    else:
        vehicle_type = "Medium Truck"

    # Road
    if "rough" in low:
        road_condition = "Rough"
    elif "mountain" in low:
        road_condition = "Mountainous"
    elif "smooth" in low:
        road_condition = "Smooth"
    elif "congested" in low or "traffic" in low:
        road_condition = "Congested"
    else:
        road_condition = "Moderate"

    # Weather
    if "hot" in low:
        weather_condition = "Hot"
    elif "cold" in low:
        weather_condition = "Cold"
    elif "rain" in low:
        weather_condition = "Rainy"
    elif "fog" in low:
        weather_condition = "Foggy"
    elif "humid" in low:
        weather_condition = "Humid"
    else:
        weather_condition = "Clear"

    # Load capacity
    load = None

    match = re.search(r'(\d{1,3}(?:,\d{3})+)', text)
    if match:
        load = int(match.group(1).replace(",", ""))

    if load is None:
        match = re.search(r'(\d+(?:\.\d+)?)\s*[kK]', text)
        if match:
            load = int(float(match.group(1)) * 1000)

    if load is None:
        numbers = [
            int(x) for x in re.findall(r'\b\d{4,5}\b', text)
            if 1000 <= int(x) <= 25000
        ]
        if numbers:
            load = max(numbers)

    if load is None:
        load = 10000

    return {
        "project_name": "New Project",
        "vehicle_type": vehicle_type,
        "min_load_capacity": load,
        "road_condition": road_condition,
        "weather_condition": weather_condition,
        "_source": "Local fallback"
    }


def parse_project_request(text):
    client = gemini_client()

    # Try Gemini first
    if client is not None:

        prompt = f"""
You are FleetOps AI.

Convert the user's request into JSON only.

Allowed vehicle types:
{VEHICLE_TYPES}

Allowed road conditions:
{ROAD_TYPES}

Allowed weather conditions:
{WEATHER_TYPES}

Return:
{{
  "project_name": "string",
  "vehicle_type": "string",
  "min_load_capacity": 15000,
  "road_condition": "string",
  "weather_condition": "string"
}}

User request:
{text}
"""

        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response.text)
            result["min_load_capacity"] = int(
                result["min_load_capacity"]
            )
            result["_source"] = "Gemini AI"

            return result

        except Exception:
            pass

    # Automatic fallback
    return local_project_parser(text)
def detect_agent_intent(text):
    low = text.lower()

    # Check direct fleet questions FIRST
    if any(k in low for k in ["maintenance", "repair", "attention"]):
        return "maintenance"

    if any(k in low for k in ["fuel", "consumption", "efficient"]):
        return "fuel"

    if any(k in low for k in ["anomal", "failure", "risk"]):
        return "risk"

    # Project matching only when the request is actually about assignment
    if any(k in low for k in [
        "project",
        "assign",
        "recommend",
        "match",
        "truck",
        "load capacity",
        "road",
        "weather"
    ]):
        return "project_match"

    return "summary"

def agent_response(data, text):
    intent = detect_agent_intent(text)

    if intent == "project_match":
        req = parse_project_request(text)
        m, eligible = match_project(
            data,
            req["vehicle_type"],
            req["min_load_capacity"],
            req["road_condition"],
            req["weather_condition"],
            5,
        )
        m = add_reason(m, req)
        if m.empty:
            return {
                "type": "text",
                "content": "I could not find eligible vehicles for those requirements."
            }
        top = m.iloc[0]
        message = (
            f"**Top recommendation: {top['Make_and_Model']}**  \n"
            f"Vehicle ID: `{top['Vehicle_ID']}`  \n"
            f"Match score: **{top['Match_Score']:.1f}%** — {top['Classification']}  \n\n"
            f"{top['Why_This_Match']}  \n\n"
            f"I analyzed **{eligible:,} eligible vehicles** before ranking the top matches."
        )
        return {"type": "match", "content": message, "table": m, "requirements": req}

    if intent == "maintenance":
        d = data.copy()
        d["maintenance_risk"] = (
            d["Maintenance_Required"].fillna(0) * 2
            + d["Failure_History"].fillna(0)
            + d["Anomalies_Detected"].fillna(0)
            + d["Downtime_Maintenance"].fillna(0) / 5
        )
        top = d.sort_values("maintenance_risk", ascending=False).head(5)
        return {
            "type": "table",
            "content": (
                "These vehicles currently show the strongest maintenance-attention signals "
                "based on maintenance requirement, failures, anomalies, and downtime."
            ),
            "table": top[[
                "Vehicle_ID","Make_and_Model","Vehicle_Type",
                "Maintenance_Severity","Failure_History",
                "Anomalies_Detected","Downtime_Maintenance"
            ]],
        }

    if intent == "fuel":
        top = data.sort_values("Fuel_Consumption").head(5)
        return {
            "type": "table",
            "content": "These are the lowest-fuel-consumption vehicles in the current fleet dataset.",
            "table": top[[
                "Vehicle_ID","Make_and_Model","Vehicle_Type",
                "Fuel_Consumption","Load_Capacity"
            ]],
        }

    if intent == "risk":
        top = data.sort_values(
            ["Anomalies_Detected","Failure_History"],
            ascending=False
        ).head(5)
        return {
            "type": "table",
            "content": "These vehicles have the strongest anomaly/failure signals in the current dataset.",
            "table": top[[
                "Vehicle_ID","Make_and_Model","Vehicle_Type",
                "Anomalies_Detected","Failure_History",
                "Maintenance_Severity"
            ]],
        }

    maint_pct = data["Maintenance_Required"].mean() * 100
    return {
        "type": "text",
        "content": (
            f"The current fleet contains **{data['Vehicle_ID'].nunique():,} vehicles** "
            f"across **{data['Vehicle_Type'].nunique()} vehicle types**. "
            f"About **{maint_pct:.1f}%** currently require maintenance, and the average "
            f"fuel-consumption value is **{data['Fuel_Consumption'].mean():.2f}**. "
            "Ask me to recommend equipment for a project, find maintenance-risk vehicles, "
            "or inspect fuel efficiency."
        )
    }


# =========================================================
# DATA SOURCE
# =========================================================
with st.sidebar:
    st.markdown("## ⚙️ FleetOps AI")
    st.caption("Operations Command Center")
    st.divider()

    dataset_path = find_dataset()
    uploaded = None

    if dataset_path:
        st.success("Fleet dataset connected")
        raw = read_csv_path(str(dataset_path))
        st.caption(dataset_path.name)
    else:
        st.markdown("### Fleet dataset")
        uploaded = st.file_uploader("Upload fleet CSV", type=["csv"])
        if uploaded is None:
            st.info("Upload the fleet CSV once to start the command center.")
            st.stop()
        raw = read_csv_upload(uploaded)

    try:
        fleet_df = prepare_data(raw)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.divider()
    if gemini_client() is not None:
        st.success("AI Agent connected")
    else:
        st.warning("AI Agent key not configured")

    st.caption(f"{len(fleet_df):,} usable records")


# =========================================================
# TOP KPIs
# =========================================================
maintenance_pct = fleet_df["Maintenance_Required"].mean() * 100
normal_pct = (fleet_df["Maintenance_Severity"] == "Normal").mean() * 100
avg_fuel = fleet_df["Fuel_Consumption"].mean()

k1, k2, k3, k4 = st.columns(4)
kpi_data = [
    ("Fleet Assets", f"{fleet_df['Vehicle_ID'].nunique():,}", "Unique assets available"),
    ("Ready / Normal", f"{normal_pct:.1f}%", "Normal maintenance severity"),
    ("Maintenance Required", f"{maintenance_pct:.1f}%", "Assets currently flagged"),
    ("Avg. Fuel Consumption", f"{avg_fuel:.2f}", "Across the current fleet"),
]
for col, (label, value, note) in zip([k1,k2,k3,k4], kpi_data):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")


# =========================================================
# TABS
# =========================================================
tab_agent, tab_match, tab_command, tab_reports, tab_method = st.tabs([
    "🤖 AI Agent",
    "🎯 Project Match",
    "📊 Command Center",
    "📝 Reports",
    "🧠 Decision Logic",
])


# -------------------------
# AGENT
# -------------------------
with tab_agent:
    left, right = st.columns([1.5, .7], gap="large")

    with left:
        st.markdown("### AI Operations Agent")
        st.caption(
            "Ask in natural language. The agent can recommend equipment, "
            "surface maintenance risks, inspect fuel efficiency, and summarize the fleet."
        )

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": (
                        "Hi — I’m FleetOps AI. Tell me what your project needs, or ask about "
                        "maintenance, fuel efficiency, anomalies, or fleet health."
                    ),
                }
            ]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "table" in msg and msg["table"] is not None:
                    st.dataframe(msg["table"], use_container_width=True, hide_index=True)

        prompt = st.chat_input("Ask FleetOps AI...")
        if prompt:
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing fleet data..."):
                    try:
                        ans = agent_response(fleet_df, prompt)
                        st.markdown(ans["content"])
                        if "table" in ans:
                            st.dataframe(ans["table"], use_container_width=True, hide_index=True)
                        st.session_state.messages.append({
                            "role":"assistant",
                            "content":ans["content"],
                            "table":ans.get("table"),
                        })
                    except Exception as e:
                        msg = "The AI service is temporarily unavailable. Try the Project Match tab."
                        st.error(msg)
                        st.caption(str(e))
                        st.session_state.messages.append({"role":"assistant","content":msg})

    with right:
        st.markdown("### Try asking")
        suggestions = [
            "I need heavy trucks for rough roads in hot weather with at least 18,000 load capacity.",
            "Which vehicles need maintenance attention?",
            "Show me the most fuel-efficient vehicles.",
            "Which vehicles have the highest anomaly risk?",
            "Give me a fleet health summary.",
        ]
        for s in suggestions:
            st.markdown(f'<div class="panel small">{s}</div><br>', unsafe_allow_html=True)


# -------------------------
# PROJECT MATCH
# -------------------------
with tab_match:
    st.markdown("### Project-to-Equipment Matching")
    st.caption("Set the operational requirements and rank the best eligible fleet assets.")

    c1, c2, c3, c4 = st.columns(4)
    vehicle = c1.selectbox("Vehicle type", VEHICLE_TYPES)
    load = c2.number_input("Minimum load capacity", min_value=0, max_value=25000, value=18000, step=500)
    road = c3.selectbox("Road condition", ROAD_TYPES)
    weather = c4.selectbox("Weather", WEATHER_TYPES)
    top_n = st.slider("Recommendations", 5, 20, 10)

    if st.button("Find Best Equipment", type="primary", use_container_width=True):
        req = {
            "project_name":"Manual Project",
            "vehicle_type":vehicle,
            "min_load_capacity":int(load),
            "road_condition":road,
            "weather_condition":weather,
        }
        m, eligible = match_project(fleet_df, vehicle, int(load), road, weather, top_n)
        m = add_reason(m, req)

        if m.empty:
            st.warning("No eligible vehicles found.")
        else:
            top = m.iloc[0]
            st.markdown(
                f"""
                <div class="best-card">
                  <div class="agent-badge">TOP RECOMMENDATION</div>
                  <div class="best-title">{top['Make_and_Model']}</div>
                  <div class="muted">Vehicle ID {top['Vehicle_ID']} • {top['Vehicle_Type']}</div>
                  <br>
                  <span class="score-pill">{top['Match_Score']:.1f}% Match</span>
                  <p style="margin-top:.8rem">{top['Why_This_Match']}</p>
                  <div class="small muted">{eligible:,} eligible vehicles analyzed</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            st.dataframe(
                m[[
                    "Vehicle_ID","Make_and_Model","Vehicle_Type","Load_Capacity",
                    "Fuel_Consumption","Road_Conditions","Weather_Conditions",
                    "Match_Score","Classification","Why_This_Match"
                ]],
                use_container_width=True,
                hide_index=True,
            )


# -------------------------
# COMMAND CENTER
# -------------------------
with tab_command:
    st.markdown("### Operations Command Center")

    left, right = st.columns(2, gap="large")

    with left:
        type_counts = fleet_df["Vehicle_Type"].value_counts().reset_index()
        type_counts.columns = ["Vehicle Type","Count"]
        fig = px.bar(type_counts, x="Vehicle Type", y="Count", title="Fleet by Vehicle Type")
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        sev = fleet_df["Maintenance_Severity"].value_counts().reset_index()
        sev.columns = ["Severity","Count"]
        fig2 = px.pie(sev, names="Severity", values="Count", hole=.55, title="Maintenance Severity")
        fig2.update_layout(height=360, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    left2, right2 = st.columns(2, gap="large")

    with left2:
        fuel = (
            fleet_df.groupby("Vehicle_Type", as_index=False)["Fuel_Consumption"]
            .mean()
            .sort_values("Fuel_Consumption")
        )
        fig3 = px.bar(fuel, x="Vehicle_Type", y="Fuel_Consumption", title="Average Fuel Consumption")
        fig3.update_layout(height=350, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with right2:
        risk = (
            fleet_df.groupby("Vehicle_Type", as_index=False)[["Failure_History","Anomalies_Detected"]]
            .mean()
        )
        fig4 = px.scatter(
            risk,
            x="Failure_History",
            y="Anomalies_Detected",
            size="Anomalies_Detected",
            hover_name="Vehicle_Type",
            title="Failure vs. Anomaly Signals",
        )
        fig4.update_layout(height=350, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig4, use_container_width=True)


# -------------------------
# REPORTS
# -------------------------
with tab_reports:
    st.markdown("### Fleet Performance Report")
    st.caption("Generate an executive snapshot from the currently loaded fleet data.")

    if st.button("Generate Executive Report", use_container_width=True):
        maint = fleet_df["Maintenance_Required"].mean() * 100
        most_common_type = fleet_df["Vehicle_Type"].mode().iloc[0]
        highest_risk = (
            fleet_df.assign(
                Risk=fleet_df["Failure_History"].fillna(0)
                + fleet_df["Anomalies_Detected"].fillna(0)
                + fleet_df["Downtime_Maintenance"].fillna(0)/5
            )
            .sort_values("Risk", ascending=False)
            .iloc[0]
        )

        report = f"""# FleetOps AI — Executive Fleet Report

## Fleet Snapshot
- Unique vehicles: {fleet_df['Vehicle_ID'].nunique():,}
- Vehicle types: {fleet_df['Vehicle_Type'].nunique()}
- Most common vehicle type: {most_common_type}
- Maintenance required: {maint:.1f}%
- Average fuel consumption: {fleet_df['Fuel_Consumption'].mean():.2f}

## Priority Attention
Vehicle {highest_risk['Vehicle_ID']} ({highest_risk['Make_and_Model']}) has one of the strongest combined risk signals in the current dataset.

## Operational Use
FleetOps AI can rank eligible vehicles against project requirements using load capacity, maintenance health, fuel efficiency, reliability, road condition, and weather condition.
"""
        st.markdown(report)
        st.download_button(
            "Download Report",
            report,
            file_name="fleetops_executive_report.md",
            mime="text/markdown",
        )


# -------------------------
# LOGIC
# -------------------------
with tab_method:
    st.markdown("### Transparent Decision Logic")
    logic_df = pd.DataFrame({
        "Factor":[
            "Load Capacity Fit","Maintenance Health","Fuel Efficiency",
            "Reliability","Road Match","Weather Match"
        ],
        "Weight":["25%","25%","15%","15%","10%","10%"],
        "Purpose":[
            "Meets project load needs without rewarding excessive oversizing.",
            "Prioritizes vehicles with healthier maintenance status.",
            "Rewards lower fuel consumption among eligible assets.",
            "Combines failures, anomalies, and maintenance downtime.",
            "Rewards assets with matching road-condition history.",
            "Rewards assets with matching weather-condition history.",
        ]
    })
    st.dataframe(logic_df, use_container_width=True, hide_index=True)

    st.info(
        "The LLM interprets the user's request. It does not rank vehicles directly. "
        "The ranking is calculated by the Python matching engine over the fleet dataset."
    )
