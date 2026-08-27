# Driver Drowsiness Detection — Fleet Safety Dashboard

A business-facing Streamlit dashboard built on top of
`Drowsiness_DL_Corrected.ipynb`. It turns the notebook's model metrics and
fatigue-fusion outputs into something a safety/operations stakeholder can
read at a glance — no notebook or code required.

## What it shows

| Tab | Audience question it answers |
|---|---|
| 🏠 Executive Summary | "What does this system do, and can I trust it?" |
| 📊 Model Performance | "Which model (Custom CNN vs MobileNetV2) should we ship?" |
| 🚦 Fatigue Risk Analysis | "How risky was this monitoring session, and when did it escalate?" |
| 🎥 Live Detection Demo | "Show me it working on a real image." |
| ⚠️ Limitations & Roadmap | "What could go wrong in the field, and what's next?" |

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens with **sample/demo data** clearly labeled as such, so it's
fully browsable on its own.

## Connecting your real notebook outputs

In Step 8.5 and Step 9–10 of the notebook, these files get written to
`PROJECT_ROOT / "results"`:

- `model_evaluation_results.csv`
- `training_time_comparison.csv`
- `fatigue_fusion_results.csv`
- `fatigue_progression.csv`
- `fatigue_transitions.csv`

Copy that `results/` folder next to `app.py` (or point the sidebar
"Results folder path" field at it) and the dashboard automatically swaps
the demo numbers for your real ones — the "showing sample data" banner
disappears once real files are detected.

## Enabling the Live Detection Demo

The notebook saves four trained models in Step 8.5:

- `eye_custom_cnn.keras`
- `eye_mobilenetv2.keras`
- `mouth_custom_cnn.keras`
- `mouth_mobilenetv2.keras`

Upload the eye and mouth `MobileNetV2` versions (the best-performing models
per the notebook's own comparison) via the sidebar uploaders, then upload a
cropped eye image and a cropped mouth image in the **Live Detection Demo**
tab and click **Run Fatigue Detection**. Without TensorFlow installed or
without models uploaded, the tab still works in a clearly-labeled preview
mode so the UI/UX can be reviewed independent of the ML environment.

## Notes

- Fatigue fusion logic mirrors the notebook exactly: **Closed eyes →
  Severe Fatigue**, **Open eyes + Yawn → Mild Fatigue**, otherwise **Alert**.
- Image preprocessing mirrors the notebook's `load_raw_image`: resize to
  224×224, keep values in 0–255 float32 — normalization is baked into the
  saved Keras models themselves, so no extra scaling happens in the app.
