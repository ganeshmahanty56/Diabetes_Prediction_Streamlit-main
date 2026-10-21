import html
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Diabetes Risk Intelligence Engine",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# ENHANCED GLASSMORPHIC STYLING & CLEAN ANCHORS
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-main: #07090E;
        --glass-surface: rgba(16, 22, 34, 0.72);
        --glass-surface-hover: rgba(24, 32, 50, 0.88);
        --border-glass: rgba(255, 255, 255, 0.09);
        --border-glass-bright: rgba(99, 102, 241, 0.45);
        
        --accent-primary: #6366F1;
        --accent-primary-glow: rgba(99, 102, 241, 0.35);
        --accent-secondary: #06B6D4;
        
        --text-primary: #F8FAFC;
        --text-secondary: #CBD5E1;
        --text-muted: #94A3B8;
        
        --status-low: #10B981;
        --status-mod: #F59E0B;
        --status-high: #F43F5E;
        
        --radius-xl: 22px;
        --radius-lg: 16px;
        --radius-md: 12px;
        --radius-sm: 8px;
        
        --ease-spring: cubic-bezier(0.16, 1, 0.3, 1);
        --dur-smooth: 220ms;
    }

    /* ===== REMOVE ALL HEADER LINK ICONS (🔗) ===== */
    a.header-anchor,
    [data-testid="stHeaderActionElements"],
    .css-15zrgzn, .e10yg2by1,
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }

    /* Base Layout */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary);
        background-color: var(--bg-main);
        -webkit-font-smoothing: antialiased;
    }

    .stApp {
        background: 
            radial-gradient(1100px 650px at 10% -5%, rgba(99, 102, 241, 0.16), transparent 65%),
            radial-gradient(900px 550px at 90% 12%, rgba(6, 182, 212, 0.13), transparent 60%),
            radial-gradient(750px 500px at 50% 100%, rgba(168, 85, 247, 0.10), transparent 70%),
            #07090E !important;
        background-attachment: fixed !important;
    }

.block-container {
        max-width: 1240px;
        padding-top: 3.5rem !important;
        padding-bottom: 4rem !important;
        padding-left: 2.2rem !important;
        padding-right: 2.2rem !important;
        margin: 0 auto;
    }

    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }

    h1, h2, h3, h4 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text-primary);
    }

    /* Hero */
    .hero-glass {
        position: relative;
        width: 100%;
        box-sizing: border-box;
        padding: 2.2rem 2.4rem;
        border-radius: var(--radius-xl);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.015) 100%);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid var(--border-glass);
        box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.5);
        margin: 0.8rem 0 1.8rem 0;
        display: block;
        overflow: visible;
        transition: transform var(--dur-smooth) var(--ease-spring);
    }
    .hero-glass:hover {
        transform: translateY(-2px);
    }
    .hero-title {
        margin: 0 0 0.5rem;
        font-size: clamp(2.1rem, 3.5vw, 2.85rem);
        font-weight: 800;
        line-height: 1.25;
        background: linear-gradient(100deg, #FFFFFF 15%, #C7D2FE 55%, #38BDF8 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        letter-spacing: -0.03em;
    }
    .hero-desc {
        color: var(--text-secondary);
        font-size: 1.05rem;
        max-width: 780px;
        margin: 0;
        line-height: 1.6;
    }

    /* Result Panel */
    .result-glass-card {
        padding: 1.6rem;
        border-radius: var(--radius-lg);
        background: var(--glass-surface);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid var(--border-glass);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
        margin-bottom: 1rem;
        transition: transform var(--dur-smooth) var(--ease-spring), box-shadow var(--dur-smooth) ease;
    }
    .result-glass-card:hover {
        will-change: transform;
        transform: translateY(-3px);
        box-shadow: 0 18px 40px -8px rgba(0, 0, 0, 0.6);
    }
    .section-tag {
        color: var(--text-muted);
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 1.3px;
        font-weight: 700;
    }
    .risk-title-glass {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0.2rem 0 0.4rem;
    }
    .risk-desc {
        color: var(--text-secondary);
        font-size: 0.96rem;
        margin: 0;
        line-height: 1.5;
    }
    .meta-divider {
        margin-top: 1.1rem;
        padding-top: 0.9rem;
        border-top: 1px solid var(--border-glass);
        font-size: 0.9rem;
        color: var(--text-secondary);
    }

    /* Driver Badges */
    .driver-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.38rem 0.85rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 0.25rem 0.3rem 0.25rem 0;
        cursor: default;
        transition: transform 180ms var(--ease-spring);
    }
    .driver-pill:hover {
        will-change: transform;
        transform: translateY(-2px) scale(1.04);
    }
    .pill-up {
        background: rgba(244, 63, 94, 0.14);
        color: #FDA4AF;
        border: 1px solid rgba(244, 63, 94, 0.35);
    }
    .pill-down {
        background: rgba(16, 185, 129, 0.14);
        color: #A7F3D0;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .pill-neutral {
        background: rgba(255, 255, 255, 0.05);
        color: var(--text-secondary);
        border: 1px solid var(--border-glass);
    }

    /* Recent Row */
    .recent-row {
        padding: 1rem 1.2rem;
        border-radius: var(--radius-md);
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--border-glass);
        margin: 0.55rem 0;
        transition: transform var(--dur-smooth) var(--ease-spring), border-color var(--dur-smooth) ease;
    }
    .recent-row:hover {
        will-change: transform;
        transform: translateX(4px);
        border-color: var(--border-glass-bright);
        background: rgba(255, 255, 255, 0.05);
    }

    .empty-state-glass {
        padding: 3.2rem 1.4rem;
        text-align: center;
        border-radius: var(--radius-lg);
        border: 1px dashed var(--border-glass);
        background: rgba(255, 255, 255, 0.015);
        color: var(--text-secondary);
        font-size: 0.96rem;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: var(--glass-surface) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: var(--radius-md) !important;
        padding: 0.9rem 1.1rem !important;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25) !important;
        transition: transform var(--dur-smooth) var(--ease-spring), border-color var(--dur-smooth) ease !important;
    }
    div[data-testid="stMetric"]:hover {
        will-change: transform;
        transform: translateY(-2px) !important;
        border-color: var(--border-glass-bright) !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--text-muted) !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        font-size: 0.74rem !important;
        letter-spacing: 0.8px !important;
    }
    div[data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 800 !important;
        font-size: 1.65rem !important;
    }

    /* Buttons */
    div.stButton > button, div.stDownloadButton > button {
        border-radius: var(--radius-md) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        min-height: 44px !important;
        border: 1px solid var(--border-glass) !important;
        transition: transform 180ms var(--ease-spring), box-shadow 180ms ease !important;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        will-change: transform;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.5) !important;
    }
    div.stButton > button[kind="secondary"], div.stDownloadButton > button {
        background: rgba(255, 255, 255, 0.04) !important;
        color: var(--text-primary) !important;
    }
    div.stButton > button[kind="secondary"]:hover, div.stDownloadButton > button:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: var(--border-glass-bright) !important;
        transform: translateY(-2px) !important;
    }

    /* Form container framing */
    div[data-testid="stForm"] {
        border: 1px solid var(--border-glass) !important;
        background: rgba(255, 255, 255, 0.015) !important;
        border-radius: var(--radius-lg) !important;
        padding: 1.3rem !important;
    }

    /* Tabs */
    div[data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.02) !important;
        padding: 6px !important;
        border-radius: var(--radius-lg) !important;
        border: 1px solid var(--border-glass) !important;
        gap: 6px !important;
        margin-bottom: 1.5rem !important;
    }
    button[data-baseweb="tab"] {
        border-radius: var(--radius-md) !important;
        padding: 0.52rem 1.25rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        background: transparent !important;
        border: none !important;
    }
    button[data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: var(--accent-primary) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
    }

    /* Bottom Info Card */
    .footer-band {
        margin-top: 3.5rem;
        padding: 2.2rem 2rem 1.6rem;
        border-radius: var(--radius-xl);
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.02) 0%, rgba(255, 255, 255, 0.005) 100%);
        border: 1px solid var(--border-glass);
        text-align: center;
    }
    .footer-band h4 {
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 0.78rem;
        margin-bottom: 0.5rem;
    }
    .footer-links {
        display: flex;
        justify-content: center;
        gap: 0.8rem;
        margin: 1.1rem 0;
        flex-wrap: wrap;
    }
    .footer-pill-btn {
        padding: 0.42rem 1.05rem;
        border-radius: 999px;
        background: rgba(99, 102, 241, 0.12);
        color: #C7D2FE !important;
        border: 1px solid rgba(99, 102, 241, 0.3);
        text-decoration: none !important;
        font-size: 0.86rem;
        font-weight: 600;
        transition: transform 180ms ease, background-color 180ms ease;
    }
    .footer-pill-btn:hover {
        transform: translateY(-2px);
        background: rgba(99, 102, 241, 0.22);
        color: #FFFFFF !important;
    }
    .footer-sub {
        color: var(--text-muted);
        font-size: 0.82rem;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# CONSTANTS & SESSION STATE
# =========================================================
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
MISSING_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "patient_id" not in st.session_state:
    st.session_state.patient_id = f"DM-{datetime.now().strftime('%Y%m%d%H%M%S')}"
if "name_cycle" not in st.session_state:
    st.session_state.name_cycle = 0

if "form_data" not in st.session_state:
    st.session_state.form_data = {
        "patient_name": "Patient",
        "age": 30,
        "glucose": 120,
        "blood_pressure": 70,
        "bmi": 25.0,
        "pregnancies": 0,
        "skin_thickness": 20,
        "insulin": 80,
        "dpf": 0.47,
    }

# =========================================================
# DATA + MODEL PIPELINE
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

@st.cache_resource
def train_model(X_train, y_train):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe

try:
    df = load_data()
except FileNotFoundError:
    st.error("diabetes.csv not found. Please ensure it is present in the application directory.")
    st.stop()

required = FEATURES + ["Outcome"]
if any(c not in df.columns for c in required):
    st.error("Dataset structure mismatch. Missing required columns.")
    st.stop()

X = df[FEATURES].copy()
y = df["Outcome"]
X[MISSING_COLUMNS] = X[MISSING_COLUMNS].replace(0, np.nan)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = train_model(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)

train_medians = X_train.median()
train_stds = X_train.std().replace(0, 1.0)
classifier = model.named_steps["classifier"]

# =========================================================
# COMPUTATIONAL HELPERS
# =========================================================
def classify_risk(p):
    if p < 0.33:
        return "Low"
    if p < 0.66:
        return "Moderate"
    return "High"

def risk_style(risk):
    styles = {
        "Low": {
            "color": "#10B981", 
            "bg": "rgba(16, 185, 129, 0.12)",
            "msg": "Favorable biomarker distribution. Estimated markers correlate with baseline cohorts."
        },
        "Moderate": {
            "color": "#F59E0B", 
            "bg": "rgba(245, 158, 11, 0.12)",
            "msg": "Elevated metabolic indices present. Targeted nutrition and routine screening recommended."
        },
        "High": {
            "color": "#F43F5E", 
            "bg": "rgba(244, 63, 94, 0.12)",
            "msg": "Critical predictive convergence detected. Comprehensive clinical consultation prioritized."
        },
    }
    return styles[risk]

def safe(v):
    return html.escape(str(v))

def normalized_feature_drivers(patient_row, medians, stds):
    scored = []
    for f in FEATURES:
        val = patient_row[f]
        med = medians[f]
        std = stds[f]
        if pd.isna(med) or pd.isna(std):
            continue
        raw_delta = val - med
        z_score = raw_delta / std
        direction = "up" if raw_delta > 0 else "down" if raw_delta < 0 else "neutral"
        scored.append((f, raw_delta, direction, abs(z_score)))
    scored.sort(key=lambda x: x[3], reverse=True)
    return scored[:5]

def radial_gauge(prob, risk):
    color = {"Low": "#10B981", "Moderate": "#F59E0B", "High": "#F43F5E"}[risk]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={
            "suffix": "%",
            "font": {"size": 36, "family": "Plus Jakarta Sans", "color": "#F8FAFC"}
        },
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33],   "color": "rgba(16, 185, 129, 0.16)"},
                {"range": [33, 66],  "color": "rgba(245, 158, 11, 0.16)"},
                {"range": [66, 100], "color": "rgba(244, 63, 94, 0.16)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": prob * 100,
            },
        },
        title={
            "text": "Risk Probability Index",
            "font": {"size": 13, "color": "#CBD5E1", "family": "Plus Jakarta Sans"}
        },
    ))
    fig.update_layout(
        height=275,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Plus Jakarta Sans"},
    )
    return fig

def render_risk_distribution_chart(cat_counts, total_evals):
    color_map = {"Low": "#10B981", "Moderate": "#F59E0B", "High": "#F43F5E"}
    colors = [color_map.get(cat, "#6366F1") for cat in cat_counts["Risk"]]

    fig = go.Figure()

    fig.add_trace(go.Pie(
        labels=cat_counts["Risk"],
        values=cat_counts["Count"],
        hole=0.68,
        pull=[0.04] * len(cat_counts),
        textinfo="label+percent",
        textposition="outside",
        insidetextorientation="radial",
        marker=dict(
            colors=colors,
            line=dict(color="#07090E", width=3)
        ),
        hovertemplate="<b>%{label} Risk</b><br>Evaluations: %{value}<br>Proportion: %{percent}<extra></extra>",
    ))

    fig.add_annotation(
        text=f"<b style='font-size:24px; color:#F8FAFC;'>{total_evals}</b><br><span style='font-size:12px; color:#94A3B8;'>Assessed</span>",
        x=0.5, y=0.5,
        font=dict(size=14, family="Plus Jakarta Sans"),
        showarrow=False
    )

    fig.update_layout(
        height=320,
        showlegend=False,
        margin=dict(l=25, r=25, t=25, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Plus Jakarta Sans", "color": "#CBD5E1"},
    )
    return fig

def build_html_report(result):
    style = risk_style(result["Risk"])
    pct = result["Probability"] * 100
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Diabetes Assessment Report</title>
<style>
body{{margin:0;padding:32px;background:#07090E;color:#F8FAFC;font-family:'Plus Jakarta Sans',sans-serif}}
.page{{max-width:740px;margin:auto;background:rgba(255,255,255,0.03);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.1);border-radius:20px;overflow:hidden;box-shadow:0 24px 48px rgba(0,0,0,0.6)}}
.header{{padding:32px;background:linear-gradient(135deg,#6366F1,#06B6D4);color:#fff}}
.content{{padding:28px}}
.result{{padding:20px;border-radius:14px;background:{style['bg']};color:{style['color']};border:1px solid {style['color']}44}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:20px 0}}
.item{{padding:14px;border-radius:10px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06)}}
.label{{display:block;font-size:11px;color:#94A3B8;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px}}
.footer{{padding:18px 28px;background:rgba(0,0,0,0.25);font-size:12px;color:#94A3B8;border-top:1px solid rgba(255,255,255,0.06)}}
</style></head><body>
<div class="page">
  <div class="header"><h1 style="margin:0 0 6px">Clinical Machine Learning Summary</h1><div>Patient Risk Stratification</div></div>
  <div class="content">
    <div class="result"><h2 style="margin:0 0 8px">{safe(result['Risk'])} Risk</h2>{safe(style['msg'])}</div>
    <div class="grid">
      <div class="item"><span class="label">Patient Name</span><b>{safe(result['Patient Name'])}</b></div>
      <div class="item"><span class="label">Patient Identifier</span><b>{safe(result['Patient ID'])}</b></div>
      <div class="item"><span class="label">Session Timestamp</span><b>{safe(result['Timestamp'])}</b></div>
      <div class="item"><span class="label">Probability</span><b>{pct:.1f}%</b></div>
    </div>
    <div class="grid">
      <div class="item"><span class="label">Age</span>{result['Age']}</div>
      <div class="item"><span class="label">Pregnancies</span>{result['Pregnancies']}</div>
      <div class="item"><span class="label">Glucose</span>{result['Glucose']} mg/dL</div>
      <div class="item"><span class="label">Blood Pressure</span>{result['BloodPressure']} mmHg</div>
      <div class="item"><span class="label">Skin Thickness</span>{result['SkinThickness']} mm</div>
      <div class="item"><span class="label">Insulin</span>{result['Insulin']} μU/mL</div>
      <div class="item"><span class="label">BMI</span>{result['BMI']}</div>
      <div class="item"><span class="label">Pedigree Factor</span>{result['DiabetesPedigreeFunction']}</div>
    </div>
    <p style="font-size:13px;color:#94A3B8">Hold-out Model Accuracy: {result['Accuracy']*100:.1f}% · Discriminative ROC-AUC: {result['AUC']:.3f}</p>
  </div>
  <div class="footer">Please consult with a doctor or qualified healthcare professional for proper medical evaluation.</div>
</div></body></html>"""

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero-glass">
        <div class="hero-title">Diabetes Risk Predictor</div>
        <div class="hero-desc">Please consult with a doctor or qualified healthcare professional for proper medical evaluation, diagnosis, and care.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# NAVIGATION TABS
# =========================================================
tab_pred, tab_dash, tab_lab, tab_hist = st.tabs(
    ["⚡ Live Estimate", "📊 Dashboard", "🧪 Model Lab", "🕒 History"]
)

# =========================================================
# TAB 1: LIVE ESTIMATE
# =========================================================
with tab_pred:
    col_inputs, col_result = st.columns([1.25, 1], gap="large")

    with col_inputs:
        with st.form("patient_clinical_form", border=False):
            st.markdown("### Patient Profile")
            
            p_name_col, p_id_col = st.columns([1.4, 1], gap="small")
            with p_name_col:
                f_name = st.text_input(
                    "Patient Name",
                    value=st.session_state.form_data["patient_name"],
                )
            with p_id_col:
                st.text_input("Patient ID", value=st.session_state.patient_id, disabled=True)

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            st.markdown("### Health Parameters")

            a, b = st.columns(2, gap="medium")
            with a:
                f_age = st.slider("Age", 1, 100, st.session_state.form_data["age"])
                f_glucose = st.slider("Glucose (mg/dL)", 0, 250, st.session_state.form_data["glucose"])
                f_blood_pressure = st.slider("Blood Pressure (mmHg)", 0, 150, st.session_state.form_data["blood_pressure"])
                f_bmi = st.slider("BMI", 0.0, 70.0, st.session_state.form_data["bmi"], 0.1)
            with b:
                f_pregnancies = st.slider("Pregnancies", 0, 20, st.session_state.form_data["pregnancies"])
                f_skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, st.session_state.form_data["skin_thickness"])
                f_insulin = st.slider("Insulin (μU/mL)", 0, 900, st.session_state.form_data["insulin"])
                f_dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, st.session_data if False else st.session_state.form_data["dpf"], 0.01)

            submit_calc = st.form_submit_button("Predict", use_container_width=True, type="primary")

            if submit_calc:
                st.session_state.form_data = {
                    "patient_name": f_name,
                    "age": f_age,
                    "glucose": f_glucose,
                    "blood_pressure": f_blood_pressure,
                    "bmi": f_bmi,
                    "pregnancies": f_pregnancies,
                    "skin_thickness": f_skin_thickness,
                    "insulin": f_insulin,
                    "dpf": f_dpf,
                }

    eval_row = pd.DataFrame([{
        "Pregnancies": st.session_state.form_data["pregnancies"],
        "Glucose": st.session_state.form_data["glucose"],
        "BloodPressure": st.session_state.form_data["blood_pressure"],
        "SkinThickness": st.session_state.form_data["skin_thickness"],
        "Insulin": st.session_state.form_data["insulin"],
        "BMI": st.session_state.form_data["bmi"],
        "DiabetesPedigreeFunction": st.session_state.form_data["dpf"],
        "Age": st.session_state.form_data["age"],
    }])

    prob = float(model.predict_proba(eval_row)[0][1])
    pred_class = int(model.predict(eval_row)[0])
    risk = classify_risk(prob)
    style = risk_style(risk)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with col_result:
        st.markdown(
            f"""
            <div class="result-glass-card" style="border-left: 4px solid {style['color']};">
                <div class="section-tag">Assessment Conclusion</div>
                <div class="risk-title-glass" style="color:{style['color']};">{risk} Risk Tier</div>
                <div class="risk-desc">{style['msg']}</div>
                <div class="meta-divider">
                    <b>Subject:</b> {safe(st.session_state.form_data['patient_name'])}
                    <span style="margin: 0 8px; color: var(--border-glass);">|</span>
                    <b>ID:</b> <span style="font-family:'JetBrains Mono';">{safe(st.session_state.patient_id)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            radial_gauge(prob, risk),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Probability", f"{prob*100:.1f}%")
        m2.metric("Category", risk)
        m3.metric("Model ROC-AUC", f"{roc_auc:.3f}")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("### Primary Predictive Drivers")
        st.caption("Standardized deviation ranking relative to population-normalized variation.")

        movers = normalized_feature_drivers(eval_row.iloc[0], train_medians, train_stds)
        chips = []
        for feat, delta, direction, z_score in movers:
            cls = {"up": "pill-up", "down": "pill-down", "neutral": "pill-neutral"}[direction]
            sign = "+" if delta > 0 else ""
            chips.append(f'<span class="driver-pill {cls}">{feat}: {sign}{delta:.1f} ({z_score:.1f}σ)</span>')
        st.markdown(
            " ".join(chips) or "<span class='driver-pill pill-neutral'>All indicators within 1.0σ of training medians</span>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        btn_c1, btn_c2 = st.columns([1, 1], gap="small")
        with btn_c1:
            if st.button("Save to History", use_container_width=True, type="primary"):
                result = {
                    "Patient ID": st.session_state.patient_id,
                    "Patient Name": st.session_state.form_data["patient_name"],
                    "Age": st.session_state.form_data["age"],
                    "Pregnancies": st.session_state.form_data["pregnancies"],
                    "Glucose": st.session_state.form_data["glucose"],
                    "BloodPressure": st.session_state.form_data["blood_pressure"],
                    "SkinThickness": st.session_state.form_data["skin_thickness"],
                    "Insulin": st.session_state.form_data["insulin"],
                    "BMI": st.session_state.form_data["bmi"],
                    "DiabetesPedigreeFunction": st.session_state.form_data["dpf"],
                    "Probability": prob,
                    "Risk": risk,
                    "Prediction": pred_class,
                    "Accuracy": float(accuracy),
                    "AUC": float(roc_auc),
                    "Timestamp": timestamp,
                }
                st.session_state.prediction_history.append(result)
                st.success("Record appended to active session.")
        with btn_c2:
            if st.button("New Patient ID", use_container_width=True, type="secondary"):
                st.session_state.patient_id = f"DM-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                st.session_state.name_cycle += 1
                st.rerun()

        report_html = build_html_report({
            "Patient ID": st.session_state.patient_id,
            "Patient Name": st.session_state.form_data["patient_name"],
            "Age": st.session_state.form_data["age"],
            "Pregnancies": st.session_state.form_data["pregnancies"],
            "Glucose": st.session_state.form_data["glucose"],
            "BloodPressure": st.session_state.form_data["blood_pressure"],
            "SkinThickness": st.session_state.form_data["skin_thickness"],
            "Insulin": st.session_state.form_data["insulin"],
            "BMI": st.session_state.form_data["bmi"],
            "DiabetesPedigreeFunction": st.session_state.form_data["dpf"],
            "Probability": prob,
            "Risk": risk,
            "Prediction": pred_class,
            "Accuracy": float(accuracy),
            "AUC": float(roc_auc),
            "Timestamp": timestamp,
        })
        st.download_button(
            "Download Diagnostic Document",
            data=report_html,
            file_name=f"diabetes_report_{st.session_state.patient_id}.html",
            mime="text/html",
            use_container_width=True,
        )

# =========================================================
# TAB 2: DASHBOARD
# =========================================================
with tab_dash:
    st.markdown("### Operational Analytics")

    history = st.session_state.prediction_history

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Evaluations Logged", len(history))
    d2.metric("Validation Accuracy", f"{accuracy*100:.1f}%")
    d3.metric("Discriminative ROC-AUC", f"{roc_auc:.3f}")
    d4.metric("Precision / Recall", f"{precision:.2f} / {recall:.2f}")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    if history:
        hist_df = pd.DataFrame(history)

        c_left, c_right = st.columns(2, gap="large")

        with c_left:
            st.markdown("### Risk Distribution Breakdown")
            cat_counts = hist_df["Risk"].value_counts().reset_index()
            cat_counts.columns = ["Risk", "Count"]

            fig_cat = render_risk_distribution_chart(cat_counts, len(hist_df))
            st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar": False})

        with c_right:
            st.markdown("### Probability Trajectory")
            fig_time = px.line(hist_df, x="Timestamp", y="Probability", markers=True)
            fig_time.update_traces(
                line=dict(color="#6366F1", width=3),
                marker=dict(size=8, color="#06B6D4", line=dict(width=1, color="#FFFFFF"))
            )
            fig_time.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=25, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Plus Jakarta Sans", "color": "#CBD5E1"},
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Probability"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title=""),
            )
            st.plotly_chart(fig_time, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("### Recent Patient Records")
        recent = hist_df.tail(5).iloc[::-1]
        for _, row in recent.iterrows():
            sty = risk_style(row["Risk"])
            st.markdown(
                f"""
                <div class="recent-row" style="border-left: 3px solid {sty['color']};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <b>{safe(row['Patient Name'])}</b>
                        <span style="font-family:'JetBrains Mono'; font-size:0.85rem; color:var(--text-muted);">{safe(row['Timestamp'])}</span>
                    </div>
                    <div style="margin-top: 4px; font-size: 0.92rem; color: var(--text-secondary);">
                        <span style="color:{sty['color']}; font-weight:700;">{row['Risk']} Tier</span>
                        &nbsp;•&nbsp; Risk Score: <b>{row['Probability']*100:.1f}%</b>
                        &nbsp;•&nbsp; Identifier: <span style="font-family:'JetBrains Mono';">{safe(row['Patient ID'])}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="empty-state-glass">
                No assessments logged in this browser session.<br>
                Calculate and record an evaluation from the <b>Live Estimate</b> view to populate real-time cohort distribution.
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# TAB 3: MODEL LAB
# =========================================================
with tab_lab:
    st.markdown("### Validation & Performance Benchmarks")

    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Validation Accuracy", f"{accuracy*100:.2f}%")
    l2.metric("ROC-AUC", f"{roc_auc:.3f}")
    l3.metric("F1 Score", f"{f1:.3f}")
    l4.metric("Precision / Recall", f"{precision:.2f} / {recall:.2f}")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    c_lab_left, c_lab_right = st.columns(2, gap="large")

    with c_lab_left:
        st.markdown("### Confusion Matrix")
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Non-Diabetic (0)", "Actual Diabetic (1)"],
            columns=["Pred Non-Diabetic (0)", "Pred Diabetic (1)"],
        )
        st.dataframe(cm_df, use_container_width=True)

    with c_lab_right:
        st.markdown("### ROC Diagnostic Curve")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"Model (AUC = {roc_auc:.3f})",
            line=dict(color="#6366F1", width=3),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline",
            line=dict(color="#475569", dash="dash"),
        ))
        fig_roc.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Plus Jakarta Sans", "color": "#CBD5E1"},
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="False Positive Rate"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="True Positive Rate"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})

    st.markdown("### Random Forest Gini Importance")
    importances = classifier.feature_importances_
    imp_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances}).sort_values("Importance", ascending=True)
    
    fig_imp = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        color="Importance",
        color_continuous_scale=["rgba(99,102,241,0.2)", "#6366F1", "#06B6D4"],
    )
    fig_imp.update_layout(
        height=360, showlegend=False, coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Plus Jakarta Sans", "color": "#CBD5E1"},
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})

# =========================================================
# TAB 4: HISTORY
# =========================================================
with tab_hist:
    st.markdown("### Prediction Ledger & Raw Data")

    history = st.session_state.prediction_history

    if history:
        hist_df = pd.DataFrame(history)
        st.dataframe(hist_df, use_container_width=True, height=400)
        
        col_h1, col_h2 = st.columns(2, gap="medium")
        with col_h1:
            st.download_button(
                "Export History (CSV)",
                data=hist_df.to_csv(index=False).encode("utf-8"),
                file_name="prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_h2:
            if st.button("Purge Session Cache", use_container_width=True, type="secondary"):
                st.session_state.prediction_history = []
                st.rerun()
    else:
        st.markdown(
            """
            <div class="empty-state-glass">
                Session ledger empty. Assessments computed during this run will appear here.
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# FOOTER BAND
# =========================================================
st.markdown(
    """
    <div class="footer-band">
        <div style="font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:1.5px; font-size:0.78rem; margin-bottom:0.5rem;">System Information & Clinical Advisory</div>
        <div style="max-width: 680px; margin: 0 auto; color: var(--text-secondary); font-size: 0.95rem; line-height: 1.6;">
            Please consult with a doctor or qualified healthcare professional for proper medical evaluation, diagnosis, and care. 
            Educational and demonstration use only.
        </div>
        <div class="footer-links">
            <a class="footer-pill-btn" href="https://www.cdc.gov/diabetes/risk-factors/" target="_blank" rel="noopener">CDC Risk Factors</a>
            <a class="footer-pill-btn" href="https://www.who.int/news-room/fact-sheets/detail/diabetes" target="_blank" rel="noopener">WHO Diabetes Facts</a>
        </div>
        <div class="footer-sub">
            © 2026 · Ganesh Mahanty · Built with Streamlit, Scikit-Learn. 
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)