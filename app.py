"""
Driver Drowsiness Detection - Fleet Safety Intelligence Dashboard
===================================================================
A business-facing Streamlit dashboard built on top of the deep-learning
pipeline in `Drowsiness_DL_Corrected.ipynb`.

The notebook trains two binary models (a Custom CNN and a MobileNetV2
transfer-learning model) for two tasks:
    - Eye State   : Open  vs Closed
    - Mouth State : Yawn  vs No Yawn

The two predictions are fused into a 3-level fatigue score:
    0 = Alert          (eyes open, not yawning)
    1 = Mild Fatigue   (eyes open, yawning)
    2 = Severe Fatigue (eyes closed)

This app repackages those outputs into something a fleet-safety manager,
not a data scientist, can read in 30 seconds: KPIs, model trust,
fatigue-risk trends, a live demo, and known limitations.

Run with:
    streamlit run app.py
"""

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Optional heavy dependency (TensorFlow). The dashboard must still run for
# business stakeholders even on a machine where TF isn't installed - only
# the "Live Detection Demo" tab needs it.
# ---------------------------------------------------------------------------
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# ===========================================================================
# PAGE CONFIG & THEME
# ===========================================================================
st.set_page_config(
    page_title="Driver Drowsiness Detection | Fleet Safety Dashboard",
    page_icon="Logo-PTS.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

FATIGUE_COLORS = {
    "Alert": "#2ECC71",
    "Mild Fatigue": "#F1C40F",
    "Severe Fatigue": "#E74C3C",
}
IMG_SIZE = (224, 224)
RESULTS_DIR_DEFAULT = "results"
MODELS_DIR_DEFAULT = "models"

st.markdown(
    """
    <style>
    .kpi-card {
        background: #ffffff10;
        border: 1px solid #ffffff20;
        border-radius: 12px;
        padding: 18px 20px;
    }
    .demo-banner {
        background-color: #2c2f36;
        border-left: 4px solid #F1C40F;
        padding: 10px 16px;
        border-radius: 6px;
        font-size: 0.9rem;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# DATA LOADING (real artifacts if present, otherwise clearly-labeled demo data)
# ===========================================================================


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path is not None and path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_evaluation_results(results_dir: str) -> tuple[pd.DataFrame, bool]:
    """Model_evaluation_results.csv -> per-model Accuracy/Precision/Recall/F1."""
    path = Path(results_dir) / "model_evaluation_results.csv"
    df = _read_csv_if_exists(path)
    if df is not None:
        return df, True

    # Demo fallback. Eye MobileNetV2 / Mouth MobileNetV2 numbers below match
    # the values actually produced by the reference notebook run; the
    # Custom-CNN rows are illustrative placeholders until real CSVs are supplied.
    demo = pd.DataFrame(
        {
            "Model": [
                "Eye Custom CNN",
                "Eye MobileNetV2",
                "Mouth Custom CNN",
                "Mouth MobileNetV2",
            ],
            "Accuracy": [94.10, 99.08, 68.50, 72.94],
            "Precision": [93.20, 99.08, 79.10, 84.72],
            "Recall": [92.80, 99.08, 51.30, 55.96],
            "F1-Score": [93.00, 99.08, 62.20, 67.40],
        }
    )
    return demo, False


@st.cache_data(show_spinner=False)
def load_training_time(results_dir: str) -> tuple[pd.DataFrame, bool]:
    path = Path(results_dir) / "training_time_comparison.csv"
    df = _read_csv_if_exists(path)
    if df is not None:
        return df, True
    demo = pd.DataFrame(
        {
            "Model": [
                "Eye Custom CNN",
                "Eye MobileNetV2",
                "Mouth Custom CNN",
                "Mouth MobileNetV2",
            ],
            "Training Time (seconds)": [420, 260, 410, 255],
        }
    )
    return demo, False


@st.cache_data(show_spinner=False)
def load_fusion_results(results_dir: str) -> tuple[pd.DataFrame, bool]:
    path = Path(results_dir) / "fatigue_fusion_results.csv"
    df = _read_csv_if_exists(path)
    if df is not None:
        return df, True

    # Simulated demo sequence -> clearly a stand-in for real fused predictions.
    rng = np.random.default_rng(42)
    n = 300
    # Bias fatigue upward over the sequence to make the trend tab meaningful.
    drift = np.linspace(0, 1, n)
    eye_prob = np.clip(rng.normal(0.85 - 0.35 * drift, 0.12, n), 0, 1)
    mouth_prob = np.clip(rng.normal(0.15 + 0.30 * drift, 0.15, n), 0, 1)

    rows = []
    for ep, mp in zip(eye_prob, mouth_prob):
        eye_state = "Open" if ep >= 0.5 else "Closed"
        mouth_state = "Yawn" if mp >= 0.5 else "No Yawn"
        if eye_state == "Closed":
            stage, label = 2, "Severe Fatigue"
        elif mouth_state == "Yawn":
            stage, label = 1, "Mild Fatigue"
        else:
            stage, label = 0, "Alert"
        rows.append(
            {
                "Eye Probability": ep,
                "Eye State": eye_state,
                "Mouth Probability": mp,
                "Mouth State": mouth_state,
                "Fatigue Stage": stage,
                "Fatigue Level": label,
            }
        )
    return pd.DataFrame(rows), False


@st.cache_data(show_spinner=False)
def load_progression(results_dir: str) -> tuple[pd.DataFrame, bool]:
    path = Path(results_dir) / "fatigue_progression.csv"
    df = _read_csv_if_exists(path)
    if df is not None:
        return df, True
    return pd.DataFrame(), False


@st.cache_data(show_spinner=False)
def load_transitions(results_dir: str) -> tuple[pd.DataFrame, bool]:
    path = Path(results_dir) / "fatigue_transitions.csv"
    df = _read_csv_if_exists(path)
    if df is not None:
        return df, True
    return pd.DataFrame(), False


def build_progression_from_fusion(fusion_df: pd.DataFrame, frames_per_minute: int = 10) -> pd.DataFrame:
    df = fusion_df.copy()
    df.insert(0, "Frame", np.arange(1, len(df) + 1))
    df["Time Interval"] = ((df["Frame"] - 1) // frames_per_minute) + 1
    summary = df.groupby("Time Interval", as_index=False).agg(
        Average_Fatigue_Stage=("Fatigue Stage", "mean"),
        Frames=("Frame", "count"),
    )

    def stage_to_level(stage):
        if stage < 0.5:
            return "Alert"
        elif stage < 1.5:
            return "Mild Fatigue"
        return "Severe Fatigue"

    summary["Fatigue Level"] = summary["Average_Fatigue_Stage"].apply(stage_to_level)
    return summary


def build_transitions_from_progression(progression_df: pd.DataFrame) -> pd.DataFrame:
    previous_level = None
    transitions = []
    for _, row in progression_df.iterrows():
        current_level = row["Fatigue Level"]
        current_minute = int(row["Time Interval"])
        if previous_level is not None and current_level != previous_level:
            transitions.append({"Minute": current_minute, "From": previous_level, "To": current_level})
        previous_level = current_level
    return pd.DataFrame(transitions)


# ===========================================================================
# INFERENCE HELPERS (Live Detection Demo tab)
# ===========================================================================


@st.cache_resource(show_spinner="Loading model...")
def load_keras_model(file_bytes: bytes, cache_key: str):
    if not TF_AVAILABLE:
        return None
    tmp_path = Path(f"/tmp/{cache_key}.keras")
    tmp_path.write_bytes(file_bytes)
    return tf.keras.models.load_model(tmp_path)


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Match the notebook's load_raw_image: resize to 224x224, keep 0-255 float32.
    Model-specific normalization (rescale / MobileNetV2 preprocess) is baked
    into each saved model, so no manual normalization happens here."""
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img).astype("float32")
    return np.expand_dims(arr, axis=0)


def get_eye_state(probability: float) -> str:
    return "Open" if probability >= 0.5 else "Closed"


def get_mouth_state(probability: float) -> str:
    return "Yawn" if probability >= 0.5 else "No Yawn"


def fatigue_decision(eye_state: str, mouth_state: str):
    if eye_state == "Closed":
        return 2, "Severe Fatigue"
    if mouth_state == "Yawn":
        return 1, "Mild Fatigue"
    return 0, "Alert"


# ===========================================================================
# SIDEBAR - data source controls
# ===========================================================================
st.sidebar.title("🚗 Data Sources")
st.sidebar.caption(
    "Point this at the `results/` folder produced by the notebook to replace "
    "demo numbers with your real training run."
)
results_dir = st.sidebar.text_input("Results folder path", value=RESULTS_DIR_DEFAULT)

st.sidebar.markdown("---")
st.sidebar.subheader("🎥 Live Demo Models (optional)")
st.sidebar.caption("Upload the saved `.keras` files to enable real inference in the Live Detection Demo tab.")
eye_model_file = st.sidebar.file_uploader("Eye state model (.keras)", type=["keras", "h5"], key="eye_model")
mouth_model_file = st.sidebar.file_uploader("Mouth state model (.keras)", type=["keras", "h5"], key="mouth_model")

if not TF_AVAILABLE:
    st.sidebar.warning("TensorFlow isn't installed in this environment - the Live Detection Demo will run in preview mode only.")

# ===========================================================================
# LOAD DATA
# ===========================================================================
eval_df, eval_real = load_evaluation_results(results_dir)
time_df, time_real = load_training_time(results_dir)
fusion_df, fusion_real = load_fusion_results(results_dir)

progression_df, progression_real = load_progression(results_dir)
if progression_df.empty:
    progression_df = build_progression_from_fusion(fusion_df)

transitions_df, transitions_real = load_transitions(results_dir)
if transitions_df.empty:
    transitions_df = build_transitions_from_progression(progression_df)

using_demo_data = not (eval_real and time_real and fusion_real)

# ===========================================================================
# HEADER
# ===========================================================================
st.title("🚗 Driver Drowsiness Detection — Fleet Safety Intelligence Dashboard")
st.caption(
    "AI-powered eye-closure and yawning analysis, fused into a real-time driver fatigue score. "
    "Built to make the deep-learning results usable for safety and operations teams."
)

if using_demo_data:
    st.markdown(
        '<div class="demo-banner">⚠️ <b>Showing sample/demo data.</b> '
        "Point the sidebar \"Results folder path\" at your notebook's `results/` output "
        "(model_evaluation_results.csv, training_time_comparison.csv, fatigue_fusion_results.csv) "
        "to see your real numbers.</div>",
        unsafe_allow_html=True,
    )

# ===========================================================================
# KPI ROW
# ===========================================================================
best_eye_row = eval_df[eval_df["Model"].str.contains("Eye")].sort_values("Accuracy", ascending=False).iloc[0]
best_mouth_row = eval_df[eval_df["Model"].str.contains("Mouth")].sort_values("Accuracy", ascending=False).iloc[0]

fatigue_counts = fusion_df["Fatigue Level"].value_counts().reindex(
    ["Alert", "Mild Fatigue", "Severe Fatigue"], fill_value=0
)
pct_severe = round(fatigue_counts.get("Severe Fatigue", 0) / max(len(fusion_df), 1) * 100, 1)
pct_alert = round(fatigue_counts.get("Alert", 0) / max(len(fusion_df), 1) * 100, 1)

k1, k2, k3, k4 = st.columns(4)
k1.metric("👁️ Eye-State Model Accuracy", f"{best_eye_row['Accuracy']:.1f}%", help=f"Best model: {best_eye_row['Model']}")
k2.metric("👄 Yawn-Detection Model Accuracy", f"{best_mouth_row['Accuracy']:.1f}%", help=f"Best model: {best_mouth_row['Model']}")
k3.metric("🟢 Frames Classified Alert", f"{pct_alert}%")
k4.metric("🔴 Frames Flagged Severe Fatigue", f"{pct_severe}%", delta=f"{fatigue_counts.get('Severe Fatigue', 0)} frames", delta_color="inverse")

st.markdown("---")

# ===========================================================================
# TABS
# ===========================================================================
tab_summary, tab_models, tab_risk, tab_demo, tab_limits = st.tabs(
    [
        "🏠 Executive Summary",
        "📊 Model Performance",
        "🚦 Fatigue Risk Analysis",
        "🎥 Live Detection Demo",
        "⚠️ Limitations & Roadmap",
    ]
)

# ---------------------------------------------------------------------------
# TAB 1: EXECUTIVE SUMMARY
# ---------------------------------------------------------------------------
with tab_summary:
    st.subheader("Why this matters")
    st.write(
        "Driver fatigue reduces alertness, reaction time and decision-making. This system watches two "
        "physiological signals from a driver-facing camera - **eye closure** and **yawning** - and fuses them "
        "into one fatigue score that operations teams can act on in real time."
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("#### How a frame becomes a decision")
        st.markdown(
            """
            1. **Eye-state model** scores each frame `Open` vs `Closed`.
            2. **Mouth-state model** scores each frame `Yawn` vs `No Yawn`.
            3. The two scores are **fused** into a 3-level fatigue stage:

            | Eye State | Mouth State | Fatigue Level | Suggested Action |
            |---|---|---|---|
            | Open | No Yawn | 🟢 Alert | None |
            | Open | Yawn | 🟡 Mild Fatigue | Monitor / suggest break |
            | Closed | any | 🔴 Severe Fatigue | Immediate alert |
            """
        )
    with c2:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=fatigue_counts.index,
                    values=fatigue_counts.values,
                    hole=0.55,
                    marker_colors=[FATIGUE_COLORS[l] for l in fatigue_counts.index],
                )
            ]
        )
        fig.update_layout(
            title="Fatigue Level Mix (sample/session)",
            margin=dict(t=40, b=0, l=0, r=0),
            height=300,
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Business takeaways")
    b1, b2, b3 = st.columns(3)
    b1.info(f"**Eye-closure detection is highly reliable** at {best_eye_row['Accuracy']:.1f}% accuracy - "
            "this is the strongest signal for catching the most dangerous case (microsleep).")
    b2.warning(f"**Yawn detection is the weaker link** at {best_mouth_row['Accuracy']:.1f}% accuracy - "
               "treat 'Mild Fatigue' alerts as advisory, not conclusive, until this model improves.")
    b3.error(f"**{pct_severe}% of monitored frames** in this session were flagged Severe Fatigue - "
             "these should map directly to an in-cab alert or dispatch notification.")

# ---------------------------------------------------------------------------
# TAB 2: MODEL PERFORMANCE
# ---------------------------------------------------------------------------
with tab_models:
    st.subheader("Custom CNN vs. MobileNetV2 (transfer learning)")
    st.caption("Two architectures were trained for each task. This compares them head-to-head so the choice of production model is defensible.")

    metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1-Score"] if c in eval_df.columns]
    fig_bar = px.bar(
        eval_df.melt(id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Score (%)"),
        x="Model",
        y="Score (%)",
        color="Metric",
        barmode="group",
        text_auto=".1f",
        height=450,
    )
    fig_bar.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig_bar, use_container_width=True)

    st.dataframe(eval_df.style.format({c: "{:.2f}" for c in metric_cols}), use_container_width=True)

    st.markdown("#### Which model should go into production?")
    best_per_task = []
    for task, keyword in [("Eye State", "Eye"), ("Mouth State", "Mouth")]:
        subset = eval_df[eval_df["Model"].str.contains(keyword)].sort_values("Accuracy", ascending=False)
        top = subset.iloc[0]
        best_per_task.append(
            {"Task": task, "Recommended Model": top["Model"], "Accuracy (%)": top["Accuracy"], "F1-Score (%)": top.get("F1-Score", np.nan)}
        )
    st.table(pd.DataFrame(best_per_task))

    st.markdown("#### Training cost")
    fig_time = px.bar(time_df, x="Model", y="Training Time (seconds)", color="Model", height=350)
    fig_time.update_layout(showlegend=False)
    st.plotly_chart(fig_time, use_container_width=True)
    st.caption(
        "MobileNetV2 uses frozen ImageNet weights, so it typically trains faster **and** scores higher than the "
        "Custom CNN trained from scratch - a strong case for shipping the transfer-learning models."
    )

# ---------------------------------------------------------------------------
# TAB 3: FATIGUE RISK ANALYSIS
# ---------------------------------------------------------------------------
with tab_risk:
    st.subheader("Simulated driver monitoring session")
    st.caption(
        "Frames are grouped into 1-minute windows (10 frames/minute) and averaged into a fatigue trend line - "
        "this is the view a safety manager would watch during or after a trip."
    )

    r1, r2, r3 = st.columns(3)
    r1.metric("Total Frames Analyzed", f"{len(fusion_df):,}")
    r2.metric("Alert Frames", f"{fatigue_counts.get('Alert', 0):,} ({pct_alert}%)")
    r3.metric("Severe Fatigue Frames", f"{fatigue_counts.get('Severe Fatigue', 0):,} ({pct_severe}%)")

    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(
            x=progression_df["Time Interval"],
            y=progression_df["Average_Fatigue_Stage"],
            mode="lines+markers",
            line=dict(width=3, color="#3498DB"),
            name="Average Fatigue Stage",
        )
    )
    fig_line.update_layout(
        title="Fatigue Progression Over Time",
        xaxis_title="Time Interval (minutes)",
        yaxis=dict(
            title="Fatigue Level",
            tickmode="array",
            tickvals=[0, 1, 2],
            ticktext=["Alert", "Mild Fatigue", "Severe Fatigue"],
            range=[-0.1, 2.1],
        ),
        height=420,
    )
    # Shade risk zones for a business-friendly read.
    fig_line.add_hrect(y0=-0.1, y1=0.5, fillcolor=FATIGUE_COLORS["Alert"], opacity=0.08, line_width=0)
    fig_line.add_hrect(y0=0.5, y1=1.5, fillcolor=FATIGUE_COLORS["Mild Fatigue"], opacity=0.08, line_width=0)
    fig_line.add_hrect(y0=1.5, y1=2.1, fillcolor=FATIGUE_COLORS["Severe Fatigue"], opacity=0.08, line_width=0)
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("#### Risk escalation events")
    if transitions_df.empty:
        st.success("No fatigue-level transitions detected in this session - the driver stayed in a single risk band throughout.")
    else:
        def _row_color(row):
            severity_rank = {"Alert": 0, "Mild Fatigue": 1, "Severe Fatigue": 2}
            escalating = severity_rank.get(row["To"], 0) > severity_rank.get(row["From"], 0)
            return ["background-color: #E7433622" if escalating else "background-color: #2ECC7122"] * len(row)

        st.dataframe(transitions_df.style.apply(_row_color, axis=1), use_container_width=True)
        n_escalations = sum(
            {"Alert": 0, "Mild Fatigue": 1, "Severe Fatigue": 2}.get(t, 0)
            > {"Alert": 0, "Mild Fatigue": 1, "Severe Fatigue": 2}.get(f, 0)
            for f, t in zip(transitions_df["From"], transitions_df["To"])
        )
        st.caption(f"🔺 {n_escalations} escalation event(s) toward higher fatigue detected in this session.")

# ---------------------------------------------------------------------------
# TAB 4: LIVE DETECTION DEMO
# ---------------------------------------------------------------------------
with tab_demo:
    st.subheader("Try it: upload an eye crop and a mouth crop")
    st.caption(
        "Uploads should be cropped, front-facing eye and mouth regions (224x224 works best), matching the "
        "format the models were trained on."
    )

    if not TF_AVAILABLE:
        st.info("TensorFlow isn't available in this environment, so this tab runs in **preview mode**: "
                "it shows the fusion logic and UI, but predictions are placeholders. Install `tensorflow` "
                "and re-run to enable real inference.")

    d1, d2 = st.columns(2)
    with d1:
        eye_image_file = st.file_uploader("Eye image", type=["jpg", "jpeg", "png", "bmp"], key="eye_img")
        if eye_image_file:
            st.image(eye_image_file, caption="Eye input", width=200)
    with d2:
        mouth_image_file = st.file_uploader("Mouth image", type=["jpg", "jpeg", "png", "bmp"], key="mouth_img")
        if mouth_image_file:
            st.image(mouth_image_file, caption="Mouth input", width=200)

    run_clicked = st.button("🔍 Run Fatigue Detection", type="primary", disabled=not (eye_image_file and mouth_image_file))

    if run_clicked:
        eye_model = load_keras_model(eye_model_file.getvalue(), "eye_model") if (TF_AVAILABLE and eye_model_file) else None
        mouth_model = load_keras_model(mouth_model_file.getvalue(), "mouth_model") if (TF_AVAILABLE and mouth_model_file) else None

        if eye_model is not None and mouth_model is not None:
            eye_arr = preprocess_image(Image.open(eye_image_file))
            mouth_arr = preprocess_image(Image.open(mouth_image_file))
            eye_prob = float(eye_model.predict(eye_arr, verbose=0).reshape(-1)[0])
            mouth_prob = float(mouth_model.predict(mouth_arr, verbose=0).reshape(-1)[0])
            source_note = "Live model inference"
        else:
            # Preview mode - deterministic pseudo-scores so the UI is demonstrable
            # without trained weights on hand. Clearly labeled as such.
            rng = np.random.default_rng(abs(hash((eye_image_file.name, mouth_image_file.name))) % (2**32))
            eye_prob = float(rng.uniform(0, 1))
            mouth_prob = float(rng.uniform(0, 1))
            source_note = "⚠️ Preview mode - upload both .keras models in the sidebar for real predictions"

        eye_state = get_eye_state(eye_prob)
        mouth_state = get_mouth_state(mouth_prob)
        stage, label = fatigue_decision(eye_state, mouth_state)

        st.markdown("---")
        res1, res2, res3 = st.columns(3)
        res1.metric("Eye State", eye_state, f"{eye_prob:.2f} open-probability")
        res2.metric("Mouth State", mouth_state, f"{mouth_prob:.2f} yawn-probability")

        color = FATIGUE_COLORS[label]
        res3.markdown(
            f"""
            <div class="kpi-card" style="border-color:{color}; text-align:center;">
                <div style="font-size:0.9rem; opacity:0.8;">Fatigue Verdict</div>
                <div style="font-size:1.6rem; font-weight:700; color:{color};">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(source_note)

# ---------------------------------------------------------------------------
# TAB 5: LIMITATIONS & ROADMAP
# ---------------------------------------------------------------------------
with tab_limits:
    st.subheader("Known limitations")
    st.write("Straight from the model-development notes - these are the conditions where accuracy will degrade in the field:")

    lim_col1, lim_col2 = st.columns(2)
    with lim_col1:
        st.markdown(
            """
            - Low or changing illumination (night driving, tunnels)
            - Head rotation and unusual camera angles
            - Occluded eyes or mouth (hand, mask, phone)
            """
        )
    with lim_col2:
        st.markdown(
            """
            - Glasses or reflections
            - Facial-expression variation (talking, laughing vs. yawning)
            - Mismatch between training images and real-world drivers/cameras
            """
        )

    st.markdown("#### Recommended next steps before fleet-wide rollout")
    st.markdown(
        """
        1. **Collect in-cab footage** from target vehicle cameras/lighting to close the domain gap.
        2. **Improve yawn detection** - it's currently the weakest signal; consider more yawn training data or a
           temporal (multi-frame) model instead of single-frame classification.
        3. **Add a temporal smoothing layer** (e.g. require N consecutive Severe-Fatigue frames before alerting)
           to cut down on false alarms from blinking or single occluded frames.
        4. **Pilot with driver consent and a human-in-the-loop review** before any automated intervention
           (alerts, dispatch notification, etc.).
        """
    )

st.markdown("---")
st.markdown(
    "<div style='text-align:center;'>Created by <b>Pearlraj</b></div>",
    unsafe_allow_html=True
)
