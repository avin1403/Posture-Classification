# 🧘 AI Posture Analysis System (YOLO11-Pose + Streamlit)

An intelligent, real-time posture analysis and ergonomic evaluation system powered by **Ultralytics YOLO11-Pose** and **Streamlit**.

The system detects 17 COCO body keypoints, computes biomechanical joint angles, classifies predefined sitting and standing posture patterns, evaluates ergonomic health, and provides actionable recommendations.

---

## 🌟 Key Features

1. **YOLO11-Pose Detection**:
   - Uses YOLO11 Pose models (`yolo11n-pose.pt`, `yolo11s-pose.pt`, `yolo11m-pose.pt`) for fast keypoint detection.
   - 17 COCO body keypoints (head, shoulders, elbows, wrists, hips, knees, ankles).
   - Side profile auto-detection (left vs. right vs. frontal).

2. **Predefined Posture Patterns**:
   - **Sitting Patterns**:
     - *Ideal Upright Sitting*: Neutral spine, head aligned over shoulders, hip/knee ~90°-105°.
     - *Slouching / Hunched Spine*: Forward torso lean (>18°), rounded thoracic back.
     - *Forward Head Posture ("Text Neck")*: Craniovertebral angle deviation (>26°).
     - *Reclined / Pelvic Slump*: Excessive backward lean (>125° hip angle).
     - *Desk / Seated Upper-Body Fallback*: Robust detection when lower body is hidden under a desk.
   - **Standing Patterns**:
     - *Ideal Neutral Posture*: Ear-shoulder-hip-knee-ankle aligned along plumb line.
     - *Slouched / Forward Lean*: Rounded shoulders and forward torso tilt.
     - *Asymmetric / Uneven Stance*: Lateral shoulder tilt or hip imbalance.

3. **Biomechanical Metrics & Scoring**:
   - **Neck / Craniovertebral Angle (CVA)**
   - **Torso Inclination Angle**
   - **Trunk-Hip Angle (Shoulder-Hip-Knee)**
   - **Knee Flexion Angle (Hip-Knee-Ankle)**
   - **Shoulder Level Tilt**
   - **Posture Health Score (0 - 100%)** displayed with interactive gauge.

4. **Streamlit Web Application**:
   - **Upload Image**: Drag-and-drop or select any JPG, PNG, WEBP photo.
   - **Live Camera Snapshot**: Capture real-time posture using your webcam.
   - **Interactive Overlays**: Skeleton lines, angle labels, bounding boxes, diagnostic badges.
   - **Export Capabilities**: Download annotated image and JSON diagnostic reports.
   - **Posture Patterns & Ergonomics Guide**: Interactive reference guide with ergonomic standards and tips.

---

## 🚀 Quick Start

### 1. Installation

Ensure Python 3.10+ is installed, then install requirements:

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

The web dashboard will launch in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
post/
├── app.py                   # Streamlit web application dashboard
├── posture_analyzer.py      # Core YOLO11-Pose biomechanical analysis engine
├── test_posture.py          # Unit tests for posture angles & classification
├── test_model_inference.py  # Model inference validation script
├── requirements.txt         # Project dependencies
└── README.md                # Documentation and usage guide
```

---

## 🧪 Running Tests

To run the posture angle and pattern classification test suite:

```bash
python test_posture.py
```
