"""
Traffic Violation Detection System — Streamlit Dashboard

Three tabs:
  1. Detect  — Upload image → annotated output + violation summary
  2. Violation Log — Searchable/filterable SQLite records
  3. Analytics — Charts, metrics, model performance
"""

import streamlit as st
import cv2
import json
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys
import os
import warnings

# Suppress harmless PyTorch CPU warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    HELMET_MODEL_PATH,
    SEATBELT_MODEL_PATH,
    DEFAULT_CONFIDENCE_THRESHOLD,
    STOP_LINE_Y_FRACTION,
    VIOLATION_LABELS,
    VIOLATION_TYPES,
)
from pipeline.orchestrator import run_pipeline
from pipeline.evidence import (
    annotate_image,
    save_annotated_image,
    save_to_db,
    get_records,
    get_analytics,
    init_db,
)


# ═══════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════

st.set_page_config(
    page_title="Traffic Violation Detection System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════
# Custom CSS
# ═══════════════════════════════════════════════

st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Header */
    .main-header {
        background: linear-gradient(90deg, #e74c3c, #c0392b, #e74c3c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
    }

    .sub-header {
        text-align: center;
        color: #8892b0;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Violation cards */
    .violation-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px 16px;
        margin: 6px 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .violation-card:hover {
        border-color: rgba(231, 76, 60, 0.5);
        background: rgba(255, 255, 255, 0.08);
    }

    .violation-detected {
        border-left: 4px solid #e74c3c;
        background: rgba(231, 76, 60, 0.08);
    }

    .violation-safe {
        border-left: 4px solid #2ecc71;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #e74c3c;
    }

    .metric-label {
        color: #8892b0;
        font-size: 0.85rem;
        margin-top: 4px;
    }

    /* Status indicators */
    .status-ok { color: #2ecc71; }
    .status-err { color: #e74c3c; }

    /* Plate display */
    .plate-display {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: #000;
        font-size: 1.3rem;
        font-weight: 800;
        padding: 8px 20px;
        border-radius: 8px;
        text-align: center;
        letter-spacing: 3px;
        font-family: 'Courier New', monospace;
        border: 2px solid #d35400;
        display: inline-block;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 8px 20px;
        color: #ccd6f6;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════

@st.cache_resource
def load_helmet_model():
    """Load fine-tuned YOLOv8 helmet detection model."""
    try:
        if HELMET_MODEL_PATH.exists():
            from ultralytics import YOLO
            model = YOLO(str(HELMET_MODEL_PATH))
            return model, True
    except Exception as e:
        return None, f"Error: {e}"
    return None, "Weights not found — run train/train_helmet.py first"


@st.cache_resource
def load_seatbelt_model():
    """Load fine-tuned YOLOv8 seatbelt detection model."""
    try:
        if SEATBELT_MODEL_PATH.exists():
            from ultralytics import YOLO
            model = YOLO(str(SEATBELT_MODEL_PATH))
            return model, True
    except Exception as e:
        return None, f"Error: {e}"
    return None, "Weights not found — run train/train_seatbelt.py or use API"


# ═══════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════

def render_sidebar():
    """Render the sidebar with configuration and model status."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # API Key
        api_key = st.text_input(
            "🔑 Roboflow API Key",
            value=st.session_state.get("api_key", os.environ.get("ROBOFLOW_API_KEY", "")),
            type="password",
            help="Required for red-light, wrong-side, seatbelt, and illegal parking detection",
        )
        st.session_state["api_key"] = api_key

        st.divider()

        # Detection settings
        st.markdown("### 🎚️ Detection Settings")

        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=DEFAULT_CONFIDENCE_THRESHOLD,
            step=0.05,
            help="Minimum confidence score to flag a violation",
        )
        st.session_state["conf_threshold"] = conf_threshold

        stop_line_y = st.slider(
            "Stop Line Position",
            min_value=0.3,
            max_value=0.8,
            value=STOP_LINE_Y_FRACTION,
            step=0.05,
            help="Vertical position of the stop line (fraction of image height)",
        )
        st.session_state["stop_line_y"] = stop_line_y

        st.divider()

        # Model status
        st.markdown("### 📊 Model Status")

        helmet_model, helmet_status = load_helmet_model()
        seatbelt_model, seatbelt_status = load_seatbelt_model()

        models_info = {
            "Helmet (YOLOv8)": helmet_status is True,
            "Seatbelt (YOLOv8/API)": seatbelt_status is True or bool(api_key),
            "Red Light (API)": bool(api_key),
            "Wrong Side (API)": bool(api_key),
            "Illegal Parking (API)": bool(api_key),
            "Triple Riding (Logic)": True,
            "Stop Line (Logic)": True,
            "Plate OCR": True,
        }

        for name, is_ok in models_info.items():
            icon = "✅" if is_ok else "❌"
            st.markdown(f"{icon} {name}")

        st.divider()
        st.markdown(
            "<div style='text-align:center; color:#555; font-size:0.75rem;'>"
            "Gridlock 2.0 — PS3<br>Traffic Violation Detection</div>",
            unsafe_allow_html=True,
        )

    return helmet_model, seatbelt_model


# ═══════════════════════════════════════════════
# Tab 1 — Detect
# ═══════════════════════════════════════════════

def render_detect_tab(helmet_model, seatbelt_model):
    """Upload image → run pipeline → show annotated output + violation card."""

    st.markdown("### 📸 Upload Traffic Image")
    st.markdown(
        "<p style='color:#8892b0;'>Upload one or more traffic camera images (JPG, PNG) for violation analysis.</p>",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Choose image(s)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_files:
        # Show placeholder
        st.markdown(
            "<div style='text-align:center; padding:80px 20px; border:2px dashed rgba(255,255,255,0.1); "
            "border-radius:16px; margin:20px 0;'>"
            "<p style='font-size:3rem;'>📷</p>"
            "<p style='color:#8892b0; font-size:1.1rem;'>Drop traffic images here to begin analysis</p>"
            "<p style='color:#555; font-size:0.8rem;'>Supports JPG, PNG • Multi-file upload enabled</p>"
            "<p style='color:#555; font-size:0.8rem; margin-top:10px;'>Video support: coming soon</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    models = {"helmet": helmet_model, "seatbelt": seatbelt_model}
    config = {
        "api_key": st.session_state.get("api_key", ""),
        "confidence_threshold": st.session_state.get("conf_threshold", DEFAULT_CONFIDENCE_THRESHOLD),
        "stop_line_y": st.session_state.get("stop_line_y", STOP_LINE_Y_FRACTION),
    }

    for uploaded_file in uploaded_files:
        st.markdown("---")
        st.markdown(f"#### 🔍 Analyzing: `{uploaded_file.name}`")

        with st.spinner("🔬 Analyzing image... Running 7 violation detectors + plate OCR..."):
            # Reset file pointer
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()

            # Run pipeline
            result = run_pipeline(file_bytes, models=models, config=config)

            # Get images for display
            raw_image = result.get("raw_image")
            processed_image = result.get("preprocessed_image")

            if raw_image is None:
                err_msg = result.get("results", {}).get("helmet", {}).get("extra", {}).get("error", "Unknown pipeline error")
                st.error(f"❌ Failed to process image. Error: {err_msg}")
                continue

            # Annotate
            display_image = processed_image if processed_image is not None else raw_image
            annotated = annotate_image(display_image, result)

            # Save
            annotated_path = save_annotated_image(annotated, f"annotated_{uploaded_file.name}")
            save_to_db(result, annotated_path)

        # ── Display Results ──
        col_img, col_info = st.columns([3, 2])

        with col_img:
            # Convert BGR to RGB for Streamlit display
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Annotated Output", width="stretch")

        with col_info:
            # Violation summary card
            st.markdown("##### 🚨 Violation Summary")

            for vtype in VIOLATION_TYPES:
                vresult = result["results"].get(
                    vtype.replace("_violation", "").replace("illegal_", "illegal_"),
                    {},
                )
                # Map violation type to result key
                key_map = {
                    "helmet_violation": "helmet",
                    "triple_riding": "triple_riding",
                    "seatbelt_violation": "seatbelt",
                    "red_light_violation": "red_light",
                    "wrong_side_violation": "wrong_side",
                    "stop_line_violation": "stop_line",
                    "illegal_parking": "illegal_parking",
                }
                result_key = key_map.get(vtype, vtype)
                vresult = result["results"].get(result_key, {})

                detected = vresult.get("detected", False)
                conf = vresult.get("confidence", 0.0)
                error = vresult.get("extra", {}).get("error", "")
                label = VIOLATION_LABELS.get(vtype, vtype)

                if detected:
                    st.markdown(
                        f'<div class="violation-card violation-detected">'
                        f'🔴 <strong>{label}</strong> — <span style="color:#e74c3c;">{conf:.1%}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                elif error:
                    st.markdown(
                        f'<div class="violation-card">'
                        f'⚠️ <strong>{label}</strong> — <span style="color:#f39c12; font-size:0.8rem;">{error}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="violation-card violation-safe">'
                        f'🟢 <strong>{label}</strong> — Clear'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Plate number
            st.markdown("##### 🔖 License Plate")
            plate = result.get("plate_number", "unknown")
            plate_conf = result.get("plate_confidence", 0.0)
            if plate and plate != "unknown":
                st.markdown(
                    f'<div class="plate-display">{plate}</div>'
                    f'<p style="color:#8892b0; font-size:0.8rem; margin-top:4px;">Confidence: {plate_conf:.1%}</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="color:#8892b0;">Plate not detected</div>',
                    unsafe_allow_html=True,
                )

            # Stats
            st.markdown("##### ⚡ Performance")
            st.markdown(
                f"⏱️ Inference: **{result.get('inference_time_ms', 0):.0f} ms** &nbsp; "
                f"🎯 Violations: **{result.get('violation_count', 0)}**"
            )


# ═══════════════════════════════════════════════
# Tab 2 — Violation Log
# ═══════════════════════════════════════════════

def render_log_tab():
    """Display searchable/filterable violation records from SQLite."""

    st.markdown("### 📋 Violation Log")
    st.markdown(
        "<p style='color:#8892b0;'>Browse and filter all detected violations.</p>",
        unsafe_allow_html=True,
    )

    # Filters
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        filter_types = st.multiselect(
            "Filter by violation type",
            options=list(VIOLATION_LABELS.keys()),
            format_func=lambda x: VIOLATION_LABELS.get(x, x),
        )

    with col2:
        date_range = st.date_input(
            "Date range",
            value=[],
            help="Filter records by date range",
        )

    with col3:
        plate_filter = st.text_input(
            "🔍 Plate #",
            placeholder="e.g. MH12",
        )

    # Query records
    date_from = None
    date_to = None
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        date_from = str(date_range[0])
        date_to = str(date_range[1])

    # For violation type filter, query each type separately and merge
    if filter_types:
        all_records = []
        for vtype in filter_types:
            records = get_records(
                violation_type=vtype,
                date_from=date_from,
                date_to=date_to,
                plate_number=plate_filter or None,
            )
            all_records.extend(records)
        # Deduplicate by id
        seen_ids = set()
        records = []
        for r in all_records:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                records.append(r)
    else:
        records = get_records(
            date_from=date_from,
            date_to=date_to,
            plate_number=plate_filter or None,
        )

    if not records:
        st.info("No violation records found. Upload and analyze images in the Detect tab first.")
        return

    # Display as dataframe
    df = pd.DataFrame(records)
    display_cols = ["id", "timestamp", "plate_number", "violations_detected", "violation_count", "inference_time_ms"]
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available_cols],
        hide_index=True,
    )

    # Detail expander
    st.markdown("##### 📄 Record Detail")
    record_id = st.selectbox(
        "Select a record to view details",
        options=[r["id"] for r in records],
        format_func=lambda x: f"Record #{x}",
    )

    if record_id:
        record = next((r for r in records if r["id"] == record_id), None)
        if record:
            with st.expander(f"📄 Record #{record_id} — Details", expanded=True):
                detail_col1, detail_col2 = st.columns([1, 1])

                with detail_col1:
                    # Show annotated image if available
                    ann_path = record.get("annotated_image_path", "")
                    if ann_path and Path(ann_path).exists():
                        img = cv2.imread(ann_path)
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            st.image(img_rgb, caption="Annotated Image", width="stretch")
                    else:
                        st.info("Annotated image not available")

                with detail_col2:
                    st.markdown(f"**Timestamp:** {record.get('timestamp', 'N/A')}")
                    st.markdown(f"**Plate:** {record.get('plate_number', 'unknown')}")
                    st.markdown(f"**Violations:** {record.get('violations_detected', '[]')}")
                    st.markdown(f"**Count:** {record.get('violation_count', 0)}")
                    st.markdown(f"**Inference:** {record.get('inference_time_ms', 0):.0f} ms")

                    # Raw JSON display removed for a cleaner UI


# ═══════════════════════════════════════════════
# Tab 3 — Analytics
# ═══════════════════════════════════════════════

def render_analytics_tab():
    """Charts, metrics, and model performance."""

    st.markdown("### 📊 Analytics Dashboard")

    analytics = get_analytics()

    # ── Metrics Row ──
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{analytics["total_images"]}</div>'
            f'<div class="metric-label">Images Processed</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{analytics["total_violations"]}</div>'
            f'<div class="metric-label">Total Violations</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{analytics["most_common"]}</div>'
            f'<div class="metric-label">Most Common Violation</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{analytics["avg_inference_ms"]:.0f} ms</div>'
            f'<div class="metric-label">Avg Inference Time</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Charts ──
    chart1, chart2 = st.columns(2)

    with chart1:
        st.markdown("##### 📊 Violations by Type")
        type_counts = analytics.get("type_counts", {})
        if type_counts:
            # Map keys to readable labels
            chart_data = {
                VIOLATION_LABELS.get(k, k): v
                for k, v in type_counts.items()
            }
            fig = px.bar(
                x=list(chart_data.keys()),
                y=list(chart_data.values()),
                labels={"x": "Violation Type", "y": "Count"},
                color=list(chart_data.values()),
                color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccd6f6",
                showlegend=False,
                coloraxis_showscale=False,
                xaxis_tickangle=-45,
                margin=dict(t=20, b=80),
            )
            st.plotly_chart(fig)
        else:
            st.info("No violation data yet.")

    with chart2:
        st.markdown("##### 📈 Violations Over Time")
        per_day = analytics.get("per_day", [])
        if per_day:
            df_daily = pd.DataFrame(per_day)
            fig = px.line(
                df_daily,
                x="date",
                y="count",
                labels={"date": "Date", "count": "Violations"},
                markers=True,
            )
            fig.update_traces(line_color="#e74c3c", line_width=2)
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccd6f6",
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig)
        else:
            st.info("No daily data yet.")

    # ── Model Performance Card ──
    st.markdown("---")
    st.markdown("##### 🏆 Model Performance Metrics")

    results_path = PROJECT_ROOT / "eval" / "results.md"

    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Check if it has actual metrics (not just template)
        if "mAP" in content and "N/A" not in content.split("mAP")[1][:20]:
            st.markdown(content)
        else:
            st.warning(
                "📋 Metrics template found but not yet populated. "
                "Run `python eval/evaluate.py` after training models to populate metrics."
            )
            with st.expander("View template"):
                st.markdown(content)
    else:
        st.warning(
            "📋 No evaluation results found. "
            "Run `python eval/evaluate.py` after training models to generate metrics."
        )

    # Avg inference stats from DB
    st.markdown(
        f"<div class='metric-card' style='margin-top:10px;'>"
        f"<div style='display:flex; justify-content:space-around;'>"
        f"<div><span class='metric-value'>{analytics['avg_per_image']}</span><br>"
        f"<span class='metric-label'>Avg Violations/Image</span></div>"
        f"<div><span class='metric-value'>{analytics['avg_inference_ms']:.0f} ms</span><br>"
        f"<span class='metric-label'>Avg Inference Time</span></div>"
        f"<div><span class='metric-value'>{analytics['total_images']}</span><br>"
        f"<span class='metric-label'>Total Analyzed</span></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════

def main():
    # Header
    st.markdown('<h1 class="main-header">🚦 Traffic Violation Detection System</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-Powered Automated Traffic Image Analysis • 7 Violation Types • Real-Time Detection</p>',
        unsafe_allow_html=True,
    )

    # Initialize DB
    init_db()

    # Sidebar
    helmet_model, seatbelt_model = render_sidebar()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Detect", "📋 Violation Log", "📊 Analytics"])

    with tab1:
        render_detect_tab(helmet_model, seatbelt_model)

    with tab2:
        render_log_tab()

    with tab3:
        render_analytics_tab()


if __name__ == "__main__":
    main()
