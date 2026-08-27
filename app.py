"""
Driver Drowsiness Detection - Fleet Safety Intelligence Dashboard
===================================================================

A Streamlit dashboard for the Driver Drowsiness Detection project.

Project tasks:
    1. Eye State Detection
       - Open
       - Closed

    2. Mouth State Detection
       - Yawn
       - No Yawn

    3. Fatigue Fusion
       - Alert
       - Mild Fatigue
       - Severe Fatigue

Models:
    - Custom CNN
    - MobileNetV2 Transfer Learning

The application automatically looks for project CSV and Keras model
files in the repository root and in the configured results/models folders.

Run locally:
    streamlit run app.py
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image


# ============================================================================
# OPTIONAL TENSORFLOW IMPORT
# ============================================================================

try:
    import tensorflow as tf

    TF_AVAILABLE = True

except Exception:
    tf = None
    TF_AVAILABLE = False


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Driver Drowsiness Detection | Fleet Safety Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CONSTANTS
# ============================================================================

FATIGUE_COLORS = {
    "Alert": "#2ECC71",
    "Mild Fatigue": "#F1C40F",
    "Severe Fatigue": "#E74C3C",
}

IMG_SIZE = (224, 224)

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_RESULTS_DIR = PROJECT_ROOT
DEFAULT_MODELS_DIR = PROJECT_ROOT


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown(
    """
    <style>

    .kpi-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }

    .demo-banner {
        background-color: rgba(241,196,15,0.10);
        border-left: 4px solid #F1C40F;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.92rem;
        margin-bottom: 15px;
    }

    .real-banner {
        background-color: rgba(46,204,113,0.10);
        border-left: 4px solid #2ECC71;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.92rem;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def read_csv_if_exists(path: Path):
    """
    Read a CSV file if it exists and can be loaded.
    """
    if path is None:
        return None

    try:
        if path.exists() and path.is_file():
            return pd.read_csv(path)
    except Exception:
        return None

    return None


def find_file(filename: str, search_dirs):
    """
    Search for a file in multiple directories.
    """

    for directory in search_dirs:

        if directory is None:
            continue

        directory = Path(directory)

        candidate = directory / filename

        if candidate.exists() and candidate.is_file():
            return candidate

    return None


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(show_spinner=False)
def load_evaluation_results(results_dir: str):

    search_dirs = [
        Path(results_dir),
        PROJECT_ROOT,
        PROJECT_ROOT / "results",
    ]

    path = find_file(
        "model_evaluation_results.csv",
        search_dirs,
    )

    if path is not None:

        df = read_csv_if_exists(path)

        if df is not None:
            return df, True, str(path)

    # ------------------------------------------------------------------------
    # DEMO DATA
    # ------------------------------------------------------------------------

    demo = pd.DataFrame(
        {
            "Model": [
                "Eye Custom CNN",
                "Eye MobileNetV2",
                "Mouth Custom CNN",
                "Mouth MobileNetV2",
            ],
            "Accuracy": [
                94.10,
                99.08,
                68.50,
                72.94,
            ],
            "Precision": [
                93.20,
                99.08,
                79.10,
                84.72,
            ],
            "Recall": [
                92.80,
                99.08,
                51.30,
                55.96,
            ],
            "F1-Score": [
                93.00,
                99.08,
                62.20,
                67.40,
            ],
        }
    )

    return demo, False, "Demo data"


@st.cache_data(show_spinner=False)
def load_training_time(results_dir: str):

    search_dirs = [
        Path(results_dir),
        PROJECT_ROOT,
        PROJECT_ROOT / "results",
    ]

    path = find_file(
        "training_time_comparison.csv",
        search_dirs,
    )

    if path is not None:

        df = read_csv_if_exists(path)

        if df is not None:
            return df, True, str(path)

    demo = pd.DataFrame(
        {
            "Model": [
                "Eye Custom CNN",
                "Eye MobileNetV2",
                "Mouth Custom CNN",
                "Mouth MobileNetV2",
            ],
            "Training Time (seconds)": [
                420,
                260,
                410,
                255,
            ],
        }
    )

    return demo, False, "Demo data"


@st.cache_data(show_spinner=False)
def load_fusion_results(results_dir: str):

    search_dirs = [
        Path(results_dir),
        PROJECT_ROOT,
        PROJECT_ROOT / "results",
    ]

    path = find_file(
        "fatigue_fusion_results.csv",
        search_dirs,
    )

    if path is not None:

        df = read_csv_if_exists(path)

        if df is not None:
            return df, True, str(path)

    # ------------------------------------------------------------------------
    # DEMO FUSION DATA
    # ------------------------------------------------------------------------

    rng = np.random.default_rng(42)

    n = 300

    drift = np.linspace(0, 1, n)

    eye_prob = np.clip(
        rng.normal(
            0.85 - 0.35 * drift,
            0.12,
            n,
        ),
        0,
        1,
    )

    mouth_prob = np.clip(
        rng.normal(
            0.15 + 0.30 * drift,
            0.15,
            n,
        ),
        0,
        1,
    )

    rows = []

    for ep, mp in zip(eye_prob, mouth_prob):

        eye_state = (
            "Open"
            if ep >= 0.5
            else "Closed"
        )

        mouth_state = (
            "Yawn"
            if mp >= 0.5
            else "No Yawn"
        )

        stage, label = fatigue_decision(
            eye_state,
            mouth_state,
        )

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

    return (
        pd.DataFrame(rows),
        False,
        "Demo data",
    )


@st.cache_data(show_spinner=False)
def load_progression(results_dir: str):

    search_dirs = [
        Path(results_dir),
        PROJECT_ROOT,
        PROJECT_ROOT / "results",
    ]

    path = find_file(
        "fatigue_progression.csv",
        search_dirs,
    )

    if path is not None:

        df = read_csv_if_exists(path)

        if df is not None:
            return df, True, str(path)

    return (
        pd.DataFrame(),
        False,
        "Not available",
    )


@st.cache_data(show_spinner=False)
def load_transitions(results_dir: str):

    search_dirs = [
        Path(results_dir),
        PROJECT_ROOT,
        PROJECT_ROOT / "results",
    ]

    path = find_file(
        "fatigue_transitions.csv",
        search_dirs,
    )

    if path is not None:

        df = read_csv_if_exists(path)

        if df is not None:
            return df, True, str(path)

    return (
        pd.DataFrame(),
        False,
        "Not available",
    )


# ============================================================================
# BUILD PROGRESSION
# ============================================================================

def build_progression_from_fusion(
    fusion_df: pd.DataFrame,
    frames_per_minute: int = 10,
):

    if fusion_df.empty:
        return pd.DataFrame()

    df = fusion_df.copy()

    if "Fatigue Stage" not in df.columns:
        return pd.DataFrame()

    df.insert(
        0,
        "Frame",
        np.arange(
            1,
            len(df) + 1,
        ),
    )

    df["Time Interval"] = (
        (df["Frame"] - 1)
        // frames_per_minute
    ) + 1

    summary = (
        df.groupby(
            "Time Interval",
            as_index=False,
        )
        .agg(
            Average_Fatigue_Stage=(
                "Fatigue Stage",
                "mean",
            ),
            Frames=(
                "Frame",
                "count",
            ),
        )
    )

    def stage_to_level(stage):

        if stage < 0.5:
            return "Alert"

        elif stage < 1.5:
            return "Mild Fatigue"

        return "Severe Fatigue"

    summary["Fatigue Level"] = (
        summary["Average_Fatigue_Stage"]
        .apply(stage_to_level)
    )

    return summary


# ============================================================================
# BUILD TRANSITIONS
# ============================================================================

def build_transitions_from_progression(
    progression_df: pd.DataFrame,
):

    if progression_df.empty:
        return pd.DataFrame()

    previous_level = None

    transitions = []

    for _, row in progression_df.iterrows():

        current_level = row["Fatigue Level"]

        current_minute = int(
            row["Time Interval"]
        )

        if (
            previous_level is not None
            and current_level != previous_level
        ):

            transitions.append(
                {
                    "Minute": current_minute,
                    "From": previous_level,
                    "To": current_level,
                }
            )

        previous_level = current_level

    return pd.DataFrame(transitions)


# ============================================================================
# MODEL FUNCTIONS
# ============================================================================

@st.cache_resource(show_spinner="Loading TensorFlow model...")
def load_keras_model(model_path: str):

    if not TF_AVAILABLE:
        return None

    try:

        return tf.keras.models.load_model(
            model_path
        )

    except Exception as e:

        st.error(
            f"Unable to load model: {e}"
        )

        return None


def preprocess_image(
    pil_image: Image.Image,
):

    img = (
        pil_image
        .convert("RGB")
        .resize(IMG_SIZE)
    )

    arr = np.asarray(
        img
    ).astype("float32")

    return np.expand_dims(
        arr,
        axis=0,
    )


def extract_probability(prediction):

    """
    Convert model output into a single probability.

    Supports:
        - sigmoid output
        - softmax output
        - single scalar prediction
    """

    arr = np.asarray(
        prediction
    ).reshape(-1)

    if len(arr) == 0:
        return 0.0

    if len(arr) == 1:
        return float(
            np.clip(
                arr[0],
                0,
                1,
            )
        )

    # Softmax-style output
    return float(
        np.clip(
            arr[-1],
            0,
            1,
        )
    )


def get_eye_state(
    probability: float,
):

    return (
        "Open"
        if probability >= 0.5
        else "Closed"
    )


def get_mouth_state(
    probability: float,
):

    return (
        "Yawn"
        if probability >= 0.5
        else "No Yawn"
    )


def fatigue_decision(
    eye_state: str,
    mouth_state: str,
):

    if eye_state == "Closed":

        return (
            2,
            "Severe Fatigue",
        )

    if mouth_state == "Yawn":

        return (
            1,
            "Mild Fatigue",
        )

    return (
        0,
        "Alert",
    )


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title(
    "🚗 Driver Drowsiness"
)

st.sidebar.caption(
    "Fleet Safety Intelligence Dashboard"
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "📂 Data Sources"
)

results_dir = st.sidebar.text_input(
    "Results folder path",
    value=str(DEFAULT_RESULTS_DIR),
    help=(
        "The app automatically searches this folder, "
        "the project root, and the results folder."
    ),
)

models_dir = st.sidebar.text_input(
    "Models folder path",
    value=str(DEFAULT_MODELS_DIR),
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "🎥 Live Detection"
)

if TF_AVAILABLE:

    st.sidebar.success(
        "TensorFlow available"
    )

else:

    st.sidebar.warning(
        "TensorFlow is not installed. "
        "Live model inference is unavailable."
    )


# ============================================================================
# LOAD PROJECT DATA
# ============================================================================

eval_df, eval_real, eval_source = (
    load_evaluation_results(
        results_dir
    )
)

time_df, time_real, time_source = (
    load_training_time(
        results_dir
    )
)

fusion_df, fusion_real, fusion_source = (
    load_fusion_results(
        results_dir
    )
)

progression_df, progression_real, progression_source = (
    load_progression(
        results_dir
    )
)

if progression_df.empty:

    progression_df = (
        build_progression_from_fusion(
            fusion_df
        )
    )


transitions_df, transitions_real, transitions_source = (
    load_transitions(
        results_dir
    )
)

if transitions_df.empty:

    transitions_df = (
        build_transitions_from_progression(
            progression_df
        )
    )


using_demo_data = not (
    eval_real
    and time_real
    and fusion_real
)


# ============================================================================
# HEADER
# ============================================================================

st.title(
    "🚗 Driver Drowsiness Detection"
)

st.subheader(
    "Fleet Safety Intelligence Dashboard"
)

st.caption(
    "AI-powered eye-closure and yawning analysis "
    "fused into a real-time driver fatigue score."
)


if using_demo_data:

    st.markdown(
        """
        <div class="demo-banner">
        ⚠️ <b>Demo data is being displayed.</b><br>
        Some project result CSV files were not found.
        Make sure the CSV files are available in the repository
        root or the configured results folder.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
        <div class="real-banner">
        ✅ <b>Real project results loaded.</b><br>
        The dashboard is using the result files generated by the
        Driver Drowsiness Detection project.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# DATA STATUS
# ============================================================================

with st.expander(
    "📁 Data Source Status"
):

    status_df = pd.DataFrame(
        {
            "Dataset": [
                "Model Evaluation",
                "Training Time",
                "Fatigue Fusion",
                "Fatigue Progression",
                "Fatigue Transitions",
            ],
            "Status": [
                "✅ Real"
                if eval_real
                else "⚠️ Demo",

                "✅ Real"
                if time_real
                else "⚠️ Demo",

                "✅ Real"
                if fusion_real
                else "⚠️ Demo",

                "✅ Real"
                if progression_real
                else "Generated",

                "✅ Real"
                if transitions_real
                else "Generated",
            ],
            "Source": [
                eval_source,
                time_source,
                fusion_source,
                progression_source,
                transitions_source,
            ],
        }
    )

    st.dataframe(
        status_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================================
# KPI CALCULATIONS
# ============================================================================

eye_models = eval_df[
    eval_df["Model"]
    .astype(str)
    .str.contains(
        "Eye",
        case=False,
        na=False,
    )
]

mouth_models = eval_df[
    eval_df["Model"]
    .astype(str)
    .str.contains(
        "Mouth",
        case=False,
        na=False,
    )
]


if not eye_models.empty:

    best_eye_row = (
        eye_models
        .sort_values(
            "Accuracy",
            ascending=False,
        )
        .iloc[0]
    )

else:

    best_eye_row = pd.Series(
        {
            "Model": "N/A",
            "Accuracy": 0,
        }
    )


if not mouth_models.empty:

    best_mouth_row = (
        mouth_models
        .sort_values(
            "Accuracy",
            ascending=False,
        )
        .iloc[0]
    )

else:

    best_mouth_row = pd.Series(
        {
            "Model": "N/A",
            "Accuracy": 0,
        }
    )


if "Fatigue Level" in fusion_df.columns:

    fatigue_counts = (
        fusion_df["Fatigue Level"]
        .value_counts()
        .reindex(
            [
                "Alert",
                "Mild Fatigue",
                "Severe Fatigue",
            ],
            fill_value=0,
        )
    )

else:

    fatigue_counts = pd.Series(
        {
            "Alert": 0,
            "Mild Fatigue": 0,
            "Severe Fatigue": 0,
        }
    )


total_frames = max(
    len(fusion_df),
    1,
)

pct_severe = round(
    fatigue_counts.get(
        "Severe Fatigue",
        0,
    )
    / total_frames
    * 100,
    1,
)

pct_alert = round(
    fatigue_counts.get(
        "Alert",
        0,
    )
    / total_frames
    * 100,
    1,
)


# ============================================================================
# KPI ROW
# ============================================================================

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "👁️ Eye-State Accuracy",
    f"{best_eye_row['Accuracy']:.1f}%",
)

k2.metric(
    "👄 Yawn Detection Accuracy",
    f"{best_mouth_row['Accuracy']:.1f}%",
)

k3.metric(
    "🟢 Alert Frames",
    f"{pct_alert}%",
)

k4.metric(
    "🔴 Severe Fatigue",
    f"{pct_severe}%",
)


st.markdown("---")


# ============================================================================
# TABS
# ============================================================================

(
    tab_summary,
    tab_models,
    tab_risk,
    tab_demo,
    tab_limits,
) = st.tabs(
    [
        "🏠 Executive Summary",
        "📊 Model Performance",
        "🚦 Fatigue Risk Analysis",
        "🎥 Live Detection Demo",
        "⚠️ Limitations & Roadmap",
    ]
)


# ============================================================================
# TAB 1 - EXECUTIVE SUMMARY
# ============================================================================

with tab_summary:

    st.subheader(
        "Why this matters"
    )

    st.write(
        """
        Driver fatigue can reduce alertness, reaction time and
        decision-making ability. This system uses a driver-facing
        camera to analyse eye closure and yawning and combines
        these signals into three fatigue levels.
        """
    )

    c1, c2 = st.columns(
        [2, 1]
    )

    with c1:

        st.markdown(
            "#### How a frame becomes a decision"
        )

        st.markdown(
            """
            1. **Eye-State Model** → Open / Closed
            2. **Mouth-State Model** → Yawn / No Yawn
            3. **Fusion Logic** → Fatigue Level

            | Eye | Mouth | Result |
            |---|---|---|
            | Open | No Yawn | 🟢 Alert |
            | Open | Yawn | 🟡 Mild Fatigue |
            | Closed | Any | 🔴 Severe Fatigue |
            """
        )

    with c2:

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=fatigue_counts.index,
                    values=fatigue_counts.values,
                    hole=0.55,
                    marker_colors=[
                        FATIGUE_COLORS.get(
                            level,
                            "#888888",
                        )
                        for level in fatigue_counts.index
                    ],
                )
            ]
        )

        fig.update_layout(
            title="Fatigue Level Distribution",
            height=320,
            margin=dict(
                t=50,
                b=0,
                l=0,
                r=0,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.markdown(
        "#### Business Takeaways"
    )

    b1, b2, b3 = st.columns(3)

    b1.info(
        f"""
        **Best Eye Model**

        {best_eye_row['Model']}

        Accuracy: {best_eye_row['Accuracy']:.1f}%
        """
    )

    b2.warning(
        f"""
        **Best Mouth Model**

        {best_mouth_row['Model']}

        Accuracy: {best_mouth_row['Accuracy']:.1f}%
        """
    )

    b3.error(
        f"""
        **Severe Fatigue**

        {pct_severe}% of analysed frames
        were classified as severe fatigue.
        """
    )


# ============================================================================
# TAB 2 - MODEL PERFORMANCE
# ============================================================================

with tab_models:

    st.subheader(
        "📊 Model Performance Comparison"
    )

    st.caption(
        "Comparison between Custom CNN and MobileNetV2 models."
    )

    metric_cols = [
        c
        for c in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
        ]
        if c in eval_df.columns
    ]

    if metric_cols:

        melted_df = eval_df.melt(
            id_vars="Model",
            value_vars=metric_cols,
            var_name="Metric",
            value_name="Score (%)",
        )

        fig_bar = px.bar(
            melted_df,
            x="Model",
            y="Score (%)",
            color="Metric",
            barmode="group",
            text_auto=".1f",
            height=480,
        )

        fig_bar.update_layout(
            yaxis_range=[
                0,
                100,
            ]
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True,
        )

    st.markdown(
        "#### Detailed Evaluation Results"
    )

    st.dataframe(
        eval_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "#### Recommended Model"
    )

    best_per_task = []

    for task, keyword in [
        ("Eye State", "Eye"),
        ("Mouth State", "Mouth"),
    ]:

        subset = eval_df[
            eval_df["Model"]
            .astype(str)
            .str.contains(
                keyword,
                case=False,
                na=False,
            )
        ]

        if not subset.empty:

            top = (
                subset
                .sort_values(
                    "Accuracy",
                    ascending=False,
                )
                .iloc[0]
            )

            best_per_task.append(
                {
                    "Task": task,
                    "Recommended Model": top["Model"],
                    "Accuracy (%)": top["Accuracy"],
                    "F1-Score (%)": top.get(
                        "F1-Score",
                        np.nan,
                    ),
                }
            )

    if best_per_task:

        st.table(
            pd.DataFrame(
                best_per_task
            )
        )

    st.markdown(
        "#### Training Time Comparison"
    )

    if (
        "Model" in time_df.columns
        and "Training Time (seconds)" in time_df.columns
    ):

        fig_time = px.bar(
            time_df,
            x="Model",
            y="Training Time (seconds)",
            color="Model",
            height=400,
        )

        fig_time.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            fig_time,
            use_container_width=True,
        )


# ============================================================================
# TAB 3 - FATIGUE RISK ANALYSIS
# ============================================================================

with tab_risk:

    st.subheader(
        "🚦 Fatigue Risk Analysis"
    )

    st.caption(
        "Fatigue levels are analysed across the monitoring session."
    )

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Total Frames",
        f"{len(fusion_df):,}",
    )

    r2.metric(
        "Alert Frames",
        f"{fatigue_counts.get('Alert', 0):,}",
        f"{pct_alert}%",
    )

    r3.metric(
        "Severe Fatigue Frames",
        f"{fatigue_counts.get('Severe Fatigue', 0):,}",
        f"{pct_severe}%",
    )

    # ------------------------------------------------------------------------
    # FATIGUE TREND
    # ------------------------------------------------------------------------

    if (
        not progression_df.empty
        and "Average_Fatigue_Stage" in progression_df.columns
    ):

        fig_line = go.Figure()

        fig_line.add_trace(
            go.Scatter(
                x=progression_df[
                    "Time Interval"
                ],
                y=progression_df[
                    "Average_Fatigue_Stage"
                ],
                mode="lines+markers",
                name="Average Fatigue Stage",
                line=dict(
                    width=3
                ),
            )
        )

        fig_line.update_layout(
            title="Fatigue Progression Over Time",
            xaxis_title="Time Interval",
            yaxis=dict(
                title="Fatigue Level",
                tickmode="array",
                tickvals=[
                    0,
                    1,
                    2,
                ],
                ticktext=[
                    "Alert",
                    "Mild Fatigue",
                    "Severe Fatigue",
                ],
                range=[
                    -0.1,
                    2.1,
                ],
            ),
            height=450,
        )

        fig_line.add_hrect(
            y0=-0.1,
            y1=0.5,
            fillcolor=FATIGUE_COLORS[
                "Alert"
            ],
            opacity=0.08,
            line_width=0,
        )

        fig_line.add_hrect(
            y0=0.5,
            y1=1.5,
            fillcolor=FATIGUE_COLORS[
                "Mild Fatigue"
            ],
            opacity=0.08,
            line_width=0,
        )

        fig_line.add_hrect(
            y0=1.5,
            y1=2.1,
            fillcolor=FATIGUE_COLORS[
                "Severe Fatigue"
            ],
            opacity=0.08,
            line_width=0,
        )

        st.plotly_chart(
            fig_line,
            use_container_width=True,
        )

    # ------------------------------------------------------------------------
    # TRANSITIONS
    # ------------------------------------------------------------------------

    st.markdown(
        "#### Fatigue-Level Transitions"
    )

    if transitions_df.empty:

        st.success(
            "No fatigue-level transitions detected."
        )

    else:

        st.dataframe(
            transitions_df,
            use_container_width=True,
            hide_index=True,
        )

        severity_rank = {
            "Alert": 0,
            "Mild Fatigue": 1,
            "Severe Fatigue": 2,
        }

        n_escalations = 0

        for _, row in transitions_df.iterrows():

            from_level = severity_rank.get(
                row["From"],
                0,
            )

            to_level = severity_rank.get(
                row["To"],
                0,
            )

            if to_level > from_level:
                n_escalations += 1

        st.caption(
            f"🔺 {n_escalations} escalation event(s) detected."
        )


# ============================================================================
# TAB 4 - LIVE DETECTION DEMO
# ============================================================================

with tab_demo:

    st.subheader(
        "🎥 Live Fatigue Detection"
    )

    st.caption(
        "Upload an eye crop and a mouth crop to test the trained models."
    )

    # ------------------------------------------------------------------------
    # FIND MODELS AUTOMATICALLY
    # ------------------------------------------------------------------------

    model_search_dirs = [
        Path(models_dir),
        PROJECT_ROOT,
        PROJECT_ROOT / "models",
    ]

    eye_model_path = find_file(
        "eye_mobilenetv2.keras",
        model_search_dirs,
    )

    mouth_model_path = find_file(
        "mouth_mobilenetv2.keras",
        model_search_dirs,
    )

    # ------------------------------------------------------------------------
    # MODEL STATUS
    # ------------------------------------------------------------------------

    if eye_model_path and mouth_model_path:

        st.success(
            "✅ MobileNetV2 eye and mouth models found automatically."
        )

    else:

        st.warning(
            """
            ⚠️ Trained model files were not found automatically.

            Expected files:

            - eye_mobilenetv2.keras
            - mouth_mobilenetv2.keras
            """
        )

    if not TF_AVAILABLE:

        st.info(
            """
            TensorFlow is not installed in the current environment.

            The upload interface is available, but real model
            inference requires TensorFlow.
            """
        )

    # ------------------------------------------------------------------------
    # IMAGE UPLOAD
    # ------------------------------------------------------------------------

    d1, d2 = st.columns(2)

    with d1:

        eye_image_file = st.file_uploader(
            "👁️ Upload Eye Image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "bmp",
            ],
            key="eye_img",
        )

        if eye_image_file:

            st.image(
                eye_image_file,
                caption="Eye Input",
                width=220,
            )

    with d2:

        mouth_image_file = st.file_uploader(
            "👄 Upload Mouth Image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "bmp",
            ],
            key="mouth_img",
        )

        if mouth_image_file:

            st.image(
                mouth_image_file,
                caption="Mouth Input",
                width=220,
            )

    run_clicked = st.button(
        "🔍 Run Fatigue Detection",
        type="primary",
        disabled=not (
            eye_image_file
            and mouth_image_file
        ),
    )

    # ------------------------------------------------------------------------
    # RUN DETECTION
    # ------------------------------------------------------------------------

    if run_clicked:

        if (
            TF_AVAILABLE
            and eye_model_path
            and mouth_model_path
        ):

            with st.spinner(
                "Running deep-learning inference..."
            ):

                eye_model = load_keras_model(
                    str(
                        eye_model_path
                    )
                )

                mouth_model = load_keras_model(
                    str(
                        mouth_model_path
                    )
                )

                if (
                    eye_model is None
                    or mouth_model is None
                ):

                    st.error(
                        "Unable to load one or both models."
                    )

                    st.stop()

                eye_arr = preprocess_image(
                    Image.open(
                        eye_image_file
                    )
                )

                mouth_arr = preprocess_image(
                    Image.open(
                        mouth_image_file
                    )
                )

                eye_prediction = (
                    eye_model.predict(
                        eye_arr,
                        verbose=0,
                    )
                )

                mouth_prediction = (
                    mouth_model.predict(
                        mouth_arr,
                        verbose=0,
                    )
                )

                eye_prob = (
                    extract_probability(
                        eye_prediction
                    )
                )

                mouth_prob = (
                    extract_probability(
                        mouth_prediction
                    )
                )

                source_note = (
                    "✅ Live prediction from trained MobileNetV2 models."
                )

        else:

            # ---------------------------------------------------------------
            # PREVIEW MODE
            # ---------------------------------------------------------------

            rng = np.random.default_rng(
                abs(
                    hash(
                        (
                            eye_image_file.name,
                            mouth_image_file.name,
                        )
                    )
                )
                % (2**32)
            )

            eye_prob = float(
                rng.uniform(
                    0,
                    1,
                )
            )

            mouth_prob = float(
                rng.uniform(
                    0,
                    1,
                )
            )

            source_note = (
                "⚠️ Preview mode. "
                "Real model files and TensorFlow are required."
            )

        # --------------------------------------------------------------------
        # FUSION
        # --------------------------------------------------------------------

        eye_state = get_eye_state(
            eye_prob
        )

        mouth_state = get_mouth_state(
            mouth_prob
        )

        stage, label = fatigue_decision(
            eye_state,
            mouth_state,
        )

        # --------------------------------------------------------------------
        # RESULTS
        # --------------------------------------------------------------------

        st.markdown("---")

        st.subheader(
            "Detection Result"
        )

        res1, res2, res3 = st.columns(3)

        res1.metric(
            "👁️ Eye State",
            eye_state,
            f"{eye_prob:.2%} probability",
        )

        res2.metric(
            "👄 Mouth State",
            mouth_state,
            f"{mouth_prob:.2%} probability",
        )

        color = FATIGUE_COLORS[
            label
        ]

        res3.markdown(
            f"""
            <div class="kpi-card"
                 style="border-color:{color};">

                <div style="font-size:0.9rem;">
                    Fatigue Verdict
                </div>

                <div style="
                    font-size:1.6rem;
                    font-weight:700;
                    color:{color};
                ">
                    {label}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            source_note
        )

        # --------------------------------------------------------------------
        # DECISION EXPLANATION
        # --------------------------------------------------------------------

        st.markdown(
            "#### Decision Explanation"
        )

        if label == "Alert":

            st.success(
                "Driver appears alert: eyes are open and no yawning was detected."
            )

        elif label == "Mild Fatigue":

            st.warning(
                "Yawning was detected while the eyes were open. "
                "The driver may require monitoring or a break."
            )

        else:

            st.error(
                "Closed eyes were detected. "
                "This is classified as Severe Fatigue and requires immediate attention."
            )


# ============================================================================
# TAB 5 - LIMITATIONS
# ============================================================================

with tab_limits:

    st.subheader(
        "⚠️ Known Limitations"
    )

    st.write(
        "Model performance may decrease under the following real-world conditions:"
    )

    lim_col1, lim_col2 = st.columns(2)

    with lim_col1:

        st.markdown(
            """
            - Low or changing illumination
            - Night driving
            - Tunnel environments
            - Head rotation
            - Unusual camera angles
            """
        )

    with lim_col2:

        st.markdown(
            """
            - Glasses and reflections
            - Face occlusion
            - Masks or hands
            - Talking and facial expressions
            - Differences between training and real-world cameras
            """
        )

    st.markdown(
        "#### Recommended Next Steps"
    )

    st.markdown(
        """
        1. Collect real in-cab driving footage.
        2. Increase the yawn training dataset.
        3. Improve performance under low-light conditions.
        4. Add temporal smoothing across consecutive frames.
        5. Reduce false alarms from blinking.
        6. Test across different drivers and camera positions.
        7. Conduct a controlled pilot before production deployment.
        """
    )


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

st.caption(
    "Driver Drowsiness Detection using Deep Learning | "
    "Custom CNN + MobileNetV2 | "
    "Eye Closure + Yawning Fusion"
)
st.markdown("---")
st.markdown(
    "<div style='text-align:center;'>Created by <b>Pearlraj</b></div>",
    unsafe_allow_html=True
)
