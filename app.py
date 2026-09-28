"""
Streamlit Posture Analysis Application using YOLO11-Pose.
Provides interactive image upload, camera capture, biomechanical angle calculations,
sitting vs. standing pattern classification, and ergonomic feedback.
"""

import io
import json
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import plotly.graph_objects as go

from posture_analyzer import PostureAnalyzer

# Page configuration
st.set_page_config(
    page_title="YOLO11-Pose | Ergonomic Posture Analyzer",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E88E5, #43A047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #212529;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status-badge-good {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        background-color: #d4edda;
        color: #155724;
        font-weight: 700;
        font-size: 0.95rem;
        border: 1px solid #c3e6cb;
    }
    .status-badge-warning {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        background-color: #fff3cd;
        color: #856404;
        font-weight: 700;
        font-size: 0.95rem;
        border: 1px solid #ffeeba;
    }
    .status-badge-alert {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        background-color: #f8d7da;
        color: #721c24;
        font-weight: 700;
        font-size: 0.95rem;
        border: 1px solid #f5c6cb;
    }
    .rec-box {
        background-color: #eef7ff;
        border-left: 4px solid #1e88e5;
        padding: 12px 16px;
        border-radius: 4px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_analyzer(model_name: str, conf_thresh: float, kpt_thresh: float) -> PostureAnalyzer:
    """Cache the PostureAnalyzer instance to avoid re-loading model on every interaction."""
    return PostureAnalyzer(model_name=model_name, conf_threshold=conf_thresh, kpt_threshold=kpt_thresh)


def create_gauge_chart(score: int) -> go.Figure:
    """Create a semi-circular gauge chart for Posture Health Score."""
    if score >= 80:
        bar_color = "#2ECC71"
    elif score >= 60:
        bar_color = "#F39C12"
    else:
        bar_color = "#E74C3C"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Posture Health Score", 'font': {'size': 16, 'color': "#555"}},
        number={'suffix': "%", 'font': {'size': 32, 'color': bar_color, 'weight': 'bold'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#aaa"},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#ddd",
            'steps': [
                {'range': [0, 60], 'color': "rgba(231, 76, 60, 0.15)"},
                {'range': [60, 80], 'color': "rgba(243, 156, 18, 0.15)"},
                {'range': [80, 100], 'color': "rgba(46, 204, 113, 0.15)"},
            ],
            'threshold': {
                'line': {'color': bar_color, 'width': 3},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)')
    return fig


def display_analysis_results(image_np: np.ndarray, analyses: list, annotated_img: np.ndarray):
    """Render comprehensive analysis dashboard for detected persons."""
    if not analyses:
        st.warning("⚠️ No person with sufficient pose keypoints detected. Try uploading an image with a clearer view of the body.")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_np, caption="Uploaded Image", use_container_width=True)
        return

    # Visual comparison
    st.subheader("🖼️ Visual Posture Inspection")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_np, caption="Original Image", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="YOLO11-Pose Annotated Assessment", use_container_width=True)

    # Download annotated image button
    rgb_annotated = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_annotated)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Download Annotated Image",
        data=byte_im,
        file_name="posture_analysis_annotated.jpg",
        mime="image/jpeg",
    )

    st.markdown("---")

    # Detailed report for each detected individual
    st.subheader(f"📊 Ergonomic Diagnostic Report ({len(analyses)} {'Person' if len(analyses) == 1 else 'Persons'} Detected)")

    for idx, p in enumerate(analyses, start=1):
        with st.container():
            st.markdown(f"### 👤 Person #{idx}: **{p['pattern']}**")

            # Status pill & score row
            status = p['status_level']
            if status == "GOOD":
                badge_html = f"<div class='status-badge-good'>✅ Healthy Alignment ({status})</div>"
            elif status == "WARNING":
                badge_html = f"<div class='status-badge-warning'>⚠️ Mild Postural Defect ({status})</div>"
            else:
                badge_html = f"<div class='status-badge-alert'>🚨 High Risk Posture ({status})</div>"

            top_col1, top_col2, top_col3 = st.columns([1.5, 1.5, 2])
            with top_col1:
                st.markdown(f"**State Detected:** `{p['state']}` (Conf: {int(p['state_confidence'] * 100)}%)")
                st.markdown(f"**Analyzed Side/View:** `{p['profile_view'].capitalize()}`")
                st.markdown(badge_html, unsafe_allow_html=True)
            with top_col2:
                gauge_fig = create_gauge_chart(p['score'])
                st.plotly_chart(gauge_fig, use_container_width=True)
            with top_col3:
                st.markdown("**Identified Posture Issues:**")
                if p['flags']:
                    for flag in p['flags']:
                        st.markdown(f"- 🔴 **{flag}**")
                else:
                    st.markdown("- 🟢 *No significant alignment issues detected!*")

            # Metric cards
            m = p['metrics']
            st.markdown("#### 📐 Biomechanical Joint Angles")
            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                neck_str = f"{m['neck_angle']}°" if m['neck_angle'] is not None else "N/A"
                neck_help = "Deviation from vertical line. Ideal: < 20°"
                st.metric("Neck Angle (CVA)", neck_str, help=neck_help)

            with c2:
                torso_str = f"{m['torso_angle']}°" if m['torso_angle'] is not None else "N/A"
                torso_help = "Torso inclination from vertical plumb line. Ideal: < 12°"
                st.metric("Torso Inclination", torso_str, help=torso_help)

            with c3:
                hip_str = f"{m['hip_angle']}°" if m['hip_angle'] is not None else "N/A"
                hip_help = "Angle between shoulder, hip, and knee. Sitting: ~90°-105°, Standing: ~170°-180°"
                st.metric("Trunk-Hip Angle", hip_str, help=hip_help)

            with c4:
                knee_str = f"{m['knee_angle']}°" if m['knee_angle'] is not None else "N/A"
                knee_help = "Angle between hip, knee, and ankle. Sitting: ~90°-100°, Standing: ~170°-180°"
                st.metric("Knee Flexion Angle", knee_str, help=knee_help)

            with c5:
                sh_str = f"{m['shoulder_tilt']}°" if m['shoulder_tilt'] is not None else "N/A"
                sh_help = "Frontal tilt between left and right shoulders. Ideal: < 5°"
                st.metric("Shoulder Level Tilt", sh_str, help=sh_help)

            # Recommendations
            st.markdown("#### 💡 Ergonomic Recommendations & Action Plan")
            with st.container():
                for rec in p['recommendations']:
                    st.markdown(f"👉 {rec}")

            # Export JSON report for this person
            report_data = {
                "person_id": idx,
                "state": p['state'],
                "pattern": p['pattern'],
                "status_level": p['status_level'],
                "score": p['score'],
                "view": p['profile_view'],
                "metrics": p['metrics'],
                "flags": p['flags'],
                "recommendations": p['recommendations'],
            }
            json_str = json.dumps(report_data, indent=2)
            st.download_button(
                label=f"📄 Export Person #{idx} Report (JSON)",
                data=json_str,
                file_name=f"posture_report_person_{idx}.json",
                mime="application/json",
                key=f"export_{idx}",
            )
            st.markdown("---")


def main():
    # Sidebar Configuration
    st.sidebar.title("⚙️ Model & Settings")

    model_option = st.sidebar.selectbox(
        "YOLO11 Pose Model",
        options=["yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt"],
        index=0,
        help="yolo11n-pose is lightweight and fastest. yolo11s or yolo11m provide higher precision.",
    )

    conf_thresh = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Minimum confidence threshold for person bounding box detection.",
    )

    kpt_thresh = st.sidebar.slider(
        "Keypoint Confidence",
        min_value=0.1,
        max_value=0.9,
        value=0.30,
        step=0.05,
        help="Minimum confidence required to accept a body keypoint.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Overlay Options")
    draw_angles = st.sidebar.checkbox("Draw Angle Labels", value=True)
    draw_bbox = st.sidebar.checkbox("Draw Person Bounding Box", value=True)
    draw_annotations = st.sidebar.checkbox("Draw Skeleton & Diagnostics", value=True)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Posture Classification Patterns:**\n\n"
        "• **Sitting**: Ideal Upright, Slouching / Hunched, Forward Head ('Text Neck'), Reclined / Slumping\n"
        "• **Standing**: Ideal Neutral, Slouched / Forward Lean, Uneven Stance\n\n"
        "Powered by **Ultralytics YOLO11-Pose**."
    )

    # Header
    st.markdown("<div class='main-title'>🧘 Ergonomic Posture Analysis System</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Advanced AI biomechanics detection using <b>YOLO11-Pose</b>. "
        "Upload a photo or capture a live webcam snapshot to diagnose sitting & standing posture patterns.</div>",
        unsafe_allow_html=True,
    )

    # Initialize analyzer
    analyzer = get_analyzer(model_option, conf_thresh, kpt_thresh)
    analyzer.change_model(model_option)
    analyzer.conf_threshold = conf_thresh
    analyzer.kpt_threshold = kpt_thresh

    # Main Tabs
    tab_upload, tab_camera, tab_guide = st.tabs([
        "📤 Upload Image",
        "📷 Live Camera Snapshot",
        "📖 Posture Patterns & Ergonomics Guide",
    ])

    # TAB 1: File Upload
    with tab_upload:
        col_up, col_sample = st.columns([3, 2])
        with col_up:
            uploaded_file = st.file_uploader(
                "Choose a posture photo (JPG, PNG, JPEG, WEBP)",
                type=["jpg", "jpeg", "png", "webp"],
                help="For best results, upload a side profile or full-body view of a person sitting or standing.",
            )
        with col_sample:
            st.markdown("**Or test with a sample image:**")
            import os
            sample_options = ["None"]
            sample_dir = "samples"
            if os.path.exists(sample_dir):
                available_samples = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png')) and not f.startswith('annotated_')]
                sample_options.extend(available_samples)
            selected_sample = st.selectbox("Select Sample Demo Image", options=sample_options)

        target_bgr = None

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            target_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif selected_sample != "None" and os.path.exists(os.path.join(sample_dir, selected_sample)):
            target_bgr = cv2.imread(os.path.join(sample_dir, selected_sample))

        if target_bgr is not None:
            image_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
            with st.spinner("Analyzing posture with YOLO11-Pose..."):
                annotated_bgr, analyses = analyzer.process_image(
                    target_bgr,
                    draw_annotations=draw_annotations,
                    draw_angles=draw_angles,
                    draw_bbox=draw_bbox,
                )

            display_analysis_results(image_rgb, analyses, annotated_bgr)
        else:
            st.info("👆 Upload an image or select a sample demo image above to begin posture analysis.")

    # TAB 2: Live Camera Snapshot
    with tab_camera:
        st.write("Position yourself in front of your camera (seated or standing) and click **Take Photo**.")
        camera_img = st.camera_input("Capture Posture Photo")

        if camera_img is not None:
            file_bytes = np.asarray(bytearray(camera_img.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            with st.spinner("Analyzing posture with YOLO11-Pose..."):
                annotated_bgr, analyses = analyzer.process_image(
                    image_bgr,
                    draw_annotations=draw_annotations,
                    draw_angles=draw_angles,
                    draw_bbox=draw_bbox,
                )

            display_analysis_results(image_rgb, analyses, annotated_bgr)

    # TAB 3: Guide
    with tab_guide:
        st.markdown(r"""
        ### 📚 Posture Patterns & Biomechanical Standards

        Proper posture minimizes strain on muscles and ligaments during weight-bearing activities.
        Here is how our AI assesses your alignment:

        ---

        #### 🪑 Sitting Posture Patterns
        1. **Ideal Upright Sitting**:
           - **Neck Angle**: $< 20^\circ$ (Ear aligned vertically over shoulder).
           - **Torso Angle**: $< 12^\circ$ (Neutral spine, back supported by chair).
           - **Hip & Knee Angles**: $\sim 90^\circ - 105^\circ$ (Thighs parallel to floor, feet flat).
           - **Health Score**: $90 - 100\%$ (Optimal).

        2. **Slouching / Hunched Back**:
           - **Characteristics**: Torso leaning forward $> 18^\circ$, rounded thoracic spine.
           - **Risks**: Lower back pain, disc compression, shoulder tightness.
           - **Fix**: Use lumbar support and keep shoulders retracted against chair backrest.

        3. **Forward Head Posture ("Text Neck")**:
           - **Characteristics**: Head craning forward past the vertical shoulder line ($> 26^\circ$).
           - **Risks**: Cervical spine strain, tension headaches, upper trapezius stiffness.
           - **Fix**: Raise monitor so top third of screen is at eye level; practice chin tucks.

        4. **Reclined / Pelvic Slump**:
           - **Characteristics**: Torso tilted far back with hips sliding forward ($> 125^\circ$ hip angle).
           - **Fix**: Adjust seat angle to $100^\circ - 110^\circ$ and sit all the way back against the lumbar pad.

        ---

        #### 🧍 Standing Posture Patterns
        1. **Ideal Neutral Standing**:
           - **Alignment**: Ear, shoulder, hip, knee, and ankle align along vertical plumb line.
           - **Neck Angle**: $< 20^\circ$, Torso: $< 10^\circ$.
           - **Leg Extension**: $> 165^\circ$.

        2. **Slouched Standing**:
           - **Characteristics**: Forward torso lean ($> 12^\circ$), rounded shoulders, and drooping head.
           - **Fix**: Stand tall, engage core gently, and distribute weight evenly across both feet.

        3. **Asymmetric / Uneven Stance**:
           - **Characteristics**: Significant shoulder tilt ($> 7^\circ$) or leaning onto one leg.
           - **Fix**: Distribute body weight equally across both feet; avoid single-hip resting.

        ---

        #### 💡 Top Ergonomic Tips
        - **20-20-20 Rule**: Every 20 minutes, look at an object 20 feet away for at least 20 seconds.
        - **Desk Height**: Elbows should rest naturally at approximately $90^\circ$ on the desk or armrests.
        - **Movement Micro-breaks**: Stand up, stretch, or walk for 2 minutes every 45-60 minutes.
        """)


if __name__ == "__main__":
    main()
