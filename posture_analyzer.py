"""
Posture Analysis Engine using YOLO11-Pose.
Detects body keypoints and evaluates predefined sitting and standing posture patterns,
angles (neck, torso, hip, knee, shoulders), and provides ergonomic health scoring.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np


class KeypointIndices:
    """COCO 17-Keypoint format indices."""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


# Standard COCO skeleton connections
SKELETON_CONNECTIONS = [
    # Head
    (KeypointIndices.NOSE, KeypointIndices.LEFT_EYE),
    (KeypointIndices.NOSE, KeypointIndices.RIGHT_EYE),
    (KeypointIndices.LEFT_EYE, KeypointIndices.LEFT_EAR),
    (KeypointIndices.RIGHT_EYE, KeypointIndices.RIGHT_EAR),
    # Upper Body
    (KeypointIndices.LEFT_EAR, KeypointIndices.LEFT_SHOULDER),
    (KeypointIndices.RIGHT_EAR, KeypointIndices.RIGHT_SHOULDER),
    (KeypointIndices.LEFT_SHOULDER, KeypointIndices.RIGHT_SHOULDER),
    (KeypointIndices.LEFT_SHOULDER, KeypointIndices.LEFT_ELBOW),
    (KeypointIndices.LEFT_ELBOW, KeypointIndices.LEFT_WRIST),
    (KeypointIndices.RIGHT_SHOULDER, KeypointIndices.RIGHT_ELBOW),
    (KeypointIndices.RIGHT_ELBOW, KeypointIndices.RIGHT_WRIST),
    # Torso
    (KeypointIndices.LEFT_SHOULDER, KeypointIndices.LEFT_HIP),
    (KeypointIndices.RIGHT_SHOULDER, KeypointIndices.RIGHT_HIP),
    (KeypointIndices.LEFT_HIP, KeypointIndices.RIGHT_HIP),
    # Lower Body
    (KeypointIndices.LEFT_HIP, KeypointIndices.LEFT_KNEE),
    (KeypointIndices.LEFT_KNEE, KeypointIndices.LEFT_ANKLE),
    (KeypointIndices.RIGHT_HIP, KeypointIndices.RIGHT_KNEE),
    (KeypointIndices.RIGHT_KNEE, KeypointIndices.RIGHT_ANKLE),
]


class PostureAnalyzer:
    """
    Evaluates sitting and standing posture from 2D pose keypoints.
    Calculates biomechanical angles, assesses ergonomic patterns,
    and overlays visual diagnostic feedback onto frames.
    """

    def __init__(self, model_name: str = "yolo11n-pose.pt", conf_threshold: float = 0.25, kpt_threshold: float = 0.30):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.kpt_threshold = kpt_threshold
        self._model = None

    @property
    def model(self):
        """Lazy load YOLO pose model to optimize memory and startup time."""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
        return self._model

    def change_model(self, model_name: str):
        """Switch pose model if needed (e.g. yolo11s-pose.pt or yolo11n-pose.pt)."""
        if model_name != self.model_name:
            self.model_name = model_name
            self._model = None

    @staticmethod
    def calculate_angle_3pt(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """
        Calculate inner angle ABC formed at vertex b between point a and point c in degrees [0, 180].
        """
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        dot_product = ba[0] * bc[0] + ba[1] * bc[1]
        norm_ba = math.hypot(ba[0], ba[1])
        norm_bc = math.hypot(bc[0], bc[1])
        if norm_ba == 0 or norm_bc == 0:
            return 0.0
        cosine_angle = max(-1.0, min(1.0, dot_product / (norm_ba * norm_bc)))
        return math.degrees(math.acos(cosine_angle))

    @staticmethod
    def calculate_vertical_angle(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """
        Calculates angle in degrees between vector (p1 -> p2) and the downward vertical plumb line.
        p1: top point (e.g. ear or shoulder)
        p2: bottom point (e.g. shoulder or hip)
        Returns:
            angle: absolute deviation from vertical line [0, 90] degrees.
            direction: 1 if p1 is shifted forward relative to p2, -1 if shifted backward.
        """
        dx = p1[0] - p2[0]
        dy = p2[1] - p1[1]  # positive when p1 is above p2 in image coordinates
        if dy == 0:
            return 90.0
        angle = math.degrees(math.atan2(abs(dx), abs(dy)))
        return angle

    @staticmethod
    def calculate_horizontal_angle(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """
        Calculates angle between the line connecting p1 and p2 and the horizontal axis.
        Ideal horizontal alignment (e.g. shoulder level) gives ~0 degrees.
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0:
            return 90.0
        return math.degrees(math.atan2(abs(dy), abs(dx)))

    def detect_best_profile(self, keypoints: np.ndarray, confs: np.ndarray) -> str:
        """
        Determines whether the person is best analyzed from the Left side, Right side, or Frontal view.
        Returns: 'left', 'right', or 'frontal'
        """
        left_side_indices = [
            KeypointIndices.LEFT_EAR,
            KeypointIndices.LEFT_SHOULDER,
            KeypointIndices.LEFT_HIP,
            KeypointIndices.LEFT_KNEE,
        ]
        right_side_indices = [
            KeypointIndices.RIGHT_EAR,
            KeypointIndices.RIGHT_SHOULDER,
            KeypointIndices.RIGHT_HIP,
            KeypointIndices.RIGHT_KNEE,
        ]

        left_conf = float(np.mean([confs[i] for i in left_side_indices]))
        right_conf = float(np.mean([confs[i] for i in right_side_indices]))

        # Check shoulder width vs torso height to distinguish frontal vs side view
        left_sh = keypoints[KeypointIndices.LEFT_SHOULDER]
        right_sh = keypoints[KeypointIndices.RIGHT_SHOULDER]
        shoulder_dist = math.hypot(left_sh[0] - right_sh[0], left_sh[1] - right_sh[1])

        left_hip = keypoints[KeypointIndices.LEFT_HIP]
        torso_height = abs(left_sh[1] - left_hip[1]) + 1e-5

        aspect_ratio = shoulder_dist / torso_height

        if aspect_ratio < 0.45 or abs(left_conf - right_conf) > 0.3:
            return 'left' if left_conf >= right_conf else 'right'
        else:
            return 'frontal' if (left_conf > self.kpt_threshold and right_conf > self.kpt_threshold) else ('left' if left_conf >= right_conf else 'right')

    def analyze_person(self, keypoints: np.ndarray, confs: np.ndarray, bbox: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Analyzes a single detected person's keypoints and evaluates posture.
        keypoints: np.ndarray of shape (17, 2)
        confs: np.ndarray of shape (17,)
        """
        profile = self.detect_best_profile(keypoints, confs)

        # Select representative points based on best side or average
        if profile == 'right':
            ear_idx = KeypointIndices.RIGHT_EAR
            sh_idx = KeypointIndices.RIGHT_SHOULDER
            hip_idx = KeypointIndices.RIGHT_HIP
            knee_idx = KeypointIndices.RIGHT_KNEE
            ankle_idx = KeypointIndices.RIGHT_ANKLE
        else:
            ear_idx = KeypointIndices.LEFT_EAR
            sh_idx = KeypointIndices.LEFT_SHOULDER
            hip_idx = KeypointIndices.LEFT_HIP
            knee_idx = KeypointIndices.LEFT_KNEE
            ankle_idx = KeypointIndices.LEFT_ANKLE

        # Fallback to nose if ear has low confidence
        if confs[ear_idx] < self.kpt_threshold and confs[KeypointIndices.NOSE] >= self.kpt_threshold:
            head_pt = (float(keypoints[KeypointIndices.NOSE][0]), float(keypoints[KeypointIndices.NOSE][1]))
            head_conf = float(confs[KeypointIndices.NOSE])
        else:
            head_pt = (float(keypoints[ear_idx][0]), float(keypoints[ear_idx][1]))
            head_conf = float(confs[ear_idx])

        sh_pt = (float(keypoints[sh_idx][0]), float(keypoints[sh_idx][1]))
        sh_conf = float(confs[sh_idx])

        hip_pt = (float(keypoints[hip_idx][0]), float(keypoints[hip_idx][1]))
        hip_conf = float(confs[hip_idx])

        knee_pt = (float(keypoints[knee_idx][0]), float(keypoints[knee_idx][1]))
        knee_conf = float(confs[knee_idx])

        ankle_pt = (float(keypoints[ankle_idx][0]), float(keypoints[ankle_idx][1]))
        ankle_conf = float(confs[ankle_idx])

        # Measurements
        metrics: Dict[str, Optional[float]] = {}
        visibility: Dict[str, bool] = {}

        # 1. Neck / Craniovertebral Angle (deviation from vertical)
        if head_conf >= self.kpt_threshold and sh_conf >= self.kpt_threshold:
            neck_angle = self.calculate_vertical_angle(head_pt, sh_pt)
            metrics['neck_angle'] = round(neck_angle, 1)
            visibility['neck'] = True
        else:
            metrics['neck_angle'] = None
            visibility['neck'] = False

        # 2. Torso Inclination Angle (deviation from vertical)
        if sh_conf >= self.kpt_threshold and hip_conf >= self.kpt_threshold:
            torso_angle = self.calculate_vertical_angle(sh_pt, hip_pt)
            metrics['torso_angle'] = round(torso_angle, 1)
            visibility['torso'] = True
        else:
            metrics['torso_angle'] = None
            visibility['torso'] = False

        # 3. Hip / Trunk Angle (Shoulder - Hip - Knee)
        if sh_conf >= self.kpt_threshold and hip_conf >= self.kpt_threshold and knee_conf >= self.kpt_threshold:
            hip_angle = self.calculate_angle_3pt(sh_pt, hip_pt, knee_pt)
            metrics['hip_angle'] = round(hip_angle, 1)
            visibility['hip'] = True
        else:
            metrics['hip_angle'] = None
            visibility['hip'] = False

        # 4. Knee Flexion Angle (Hip - Knee - Ankle)
        if hip_conf >= self.kpt_threshold and knee_conf >= self.kpt_threshold and ankle_conf >= self.kpt_threshold:
            knee_angle = self.calculate_angle_3pt(hip_pt, knee_pt, ankle_pt)
            metrics['knee_angle'] = round(knee_angle, 1)
            visibility['knee'] = True
        else:
            metrics['knee_angle'] = None
            visibility['knee'] = False

        # 5. Shoulder Tilt (Frontal plane balance)
        l_sh_conf = confs[KeypointIndices.LEFT_SHOULDER]
        r_sh_conf = confs[KeypointIndices.RIGHT_SHOULDER]
        if l_sh_conf >= self.kpt_threshold and r_sh_conf >= self.kpt_threshold:
            l_sh = (float(keypoints[KeypointIndices.LEFT_SHOULDER][0]), float(keypoints[KeypointIndices.LEFT_SHOULDER][1]))
            r_sh = (float(keypoints[KeypointIndices.RIGHT_SHOULDER][0]), float(keypoints[KeypointIndices.RIGHT_SHOULDER][1]))
            sh_tilt = self.calculate_horizontal_angle(l_sh, r_sh)
            metrics['shoulder_tilt'] = round(sh_tilt, 1)
            visibility['shoulder_tilt'] = True
        else:
            metrics['shoulder_tilt'] = None
            visibility['shoulder_tilt'] = False

        # Classify Sitting vs Standing
        state, state_confidence = self._classify_sitting_or_standing(
            metrics, knee_conf, hip_conf, sh_pt, hip_pt, knee_pt
        )

        # Classify Posture Pattern & Ergonomic Rating
        pattern, status_level, score, flags, recommendations = self._evaluate_posture_pattern(
            state, metrics, profile
        )

        return {
            'state': state,
            'state_confidence': state_confidence,
            'pattern': pattern,
            'status_level': status_level,  # 'GOOD', 'WARNING', 'ALERT'
            'score': score,  # 0 to 100
            'metrics': metrics,
            'visibility': visibility,
            'profile_view': profile,
            'flags': flags,
            'recommendations': recommendations,
            'keypoints': keypoints,
            'confs': confs,
            'bbox': bbox,
            'points_used': {
                'head': head_pt,
                'shoulder': sh_pt,
                'hip': hip_pt,
                'knee': knee_pt,
                'ankle': ankle_pt,
            }
        }

    def _classify_sitting_or_standing(
        self,
        metrics: Dict[str, Optional[float]],
        knee_conf: float,
        hip_conf: float,
        sh_pt: Tuple[float, float],
        hip_pt: Tuple[float, float],
        knee_pt: Tuple[float, float],
    ) -> Tuple[str, float]:
        """
        Classifies state as Sitting vs Standing with fallback heuristic for desk-seated images.
        """
        knee_angle = metrics.get('knee_angle')
        hip_angle = metrics.get('hip_angle')

        # Full body available with knee and hip angles
        if knee_angle is not None and hip_angle is not None:
            if knee_angle < 135 and hip_angle < 140:
                return 'Sitting', 0.95
            elif knee_angle >= 145 and hip_angle >= 145:
                return 'Standing', 0.95
            elif knee_angle < 130:
                return 'Sitting', 0.85
            else:
                return 'Standing', 0.80

        # Knee and Hip visible (even if ankle is occluded)
        if knee_conf >= self.kpt_threshold and hip_conf >= self.kpt_threshold:
            # In sitting, thigh is roughly horizontal (hip and knee y-coordinates close)
            # In standing, thigh is vertical (knee y is much larger than hip y)
            thigh_dx = abs(knee_pt[0] - hip_pt[0])
            thigh_dy = abs(knee_pt[1] - hip_pt[1])
            torso_dy = abs(hip_pt[1] - sh_pt[1]) + 1e-5

            if thigh_dy / torso_dy < 0.55 or (thigh_dx > thigh_dy * 0.7):
                return 'Sitting', 0.85
            else:
                return 'Standing', 0.85

        # Lower body occluded (e.g. desktop/office webcam)
        return 'Sitting (Desk / Upper Body)', 0.70

    def _evaluate_posture_pattern(
        self,
        state: str,
        metrics: Dict[str, Optional[float]],
        profile: str,
    ) -> Tuple[str, str, int, List[str], List[str]]:
        """
        Evaluates posture patterns, detects faults, calculates 0-100 score,
        and provides tailored ergonomic recommendations.
        """
        flags = []
        recommendations = []
        score = 100

        neck = metrics.get('neck_angle')
        torso = metrics.get('torso_angle')
        hip = metrics.get('hip_angle')
        sh_tilt = metrics.get('shoulder_tilt')

        # Frontal asymmetry check
        if sh_tilt is not None and sh_tilt > 7.0:
            flags.append(f"Uneven Shoulders ({sh_tilt:.1f}° tilt)")
            score -= 15
            recommendations.append("Level your shoulders; avoid leaning heavily on one armrest or side.")

        # Sitting Analysis
        if 'Sitting' in state:
            # 1. Slouching / Hunched Back check
            is_slouched = False
            if torso is not None:
                if torso > 18.0:
                    is_slouched = True
                    flags.append(f"Severe Slouch / Forward Lean ({torso:.1f}° torso lean)")
                    score -= 30
                    recommendations.append("Straighten your upper torso against the backrest with lumbar support.")
                elif torso > 12.0:
                    flags.append(f"Mild Torso Lean ({torso:.1f}°)")
                    score -= 15
                    recommendations.append("Engage your core gently to keep your spine vertically neutral.")

            # 2. Forward Head / Text Neck check
            is_forward_head = False
            if neck is not None:
                if neck > 26.0:
                    is_forward_head = True
                    flags.append(f"Excessive Forward Head / 'Text Neck' ({neck:.1f}°)")
                    score -= 30
                    recommendations.append("Elevate your monitor to eye level and perform chin tucks to relieve cervical strain.")
                elif neck > 18.0:
                    flags.append(f"Mild Forward Head Tilt ({neck:.1f}°)")
                    score -= 15
                    recommendations.append("Position your screen closer or raise font size to avoid craning your neck forward.")

            # 3. Excessive Reclining check
            is_reclined = False
            if hip is not None and hip > 125.0 and torso is not None and torso > 15.0:
                is_reclined = True
                flags.append(f"Excessive Reclining / Slumping ({hip:.1f}° hip angle)")
                score -= 20
                recommendations.append("Adjust chair backrest to 100°-110° to prevent sliding forward on the seat pan.")

            # Determine Pattern Title
            if is_slouched and is_forward_head:
                pattern = "Sitting: Slouched with Forward Head ('Tech Slouch')"
            elif is_slouched:
                pattern = "Sitting: Slouching / Hunched Spine"
            elif is_forward_head:
                pattern = "Sitting: Forward Head Posture ('Text Neck')"
            elif is_reclined:
                pattern = "Sitting: Reclined / Pelvic Slump"
            elif len(flags) > 0:
                pattern = "Sitting: Sub-Optimal Posture"
            else:
                pattern = "Sitting: Ideal Upright Posture"

        else:
            # Standing Analysis
            is_slouched_stand = False
            if torso is not None and torso > 12.0:
                is_slouched_stand = True
                flags.append(f"Forward Torso Lean ({torso:.1f}°)")
                score -= 25
                recommendations.append("Align your shoulders directly above your hips along the vertical plumb line.")

            is_forward_head_stand = False
            if neck is not None and neck > 22.0:
                is_forward_head_stand = True
                flags.append(f"Forward Head Tilt ({neck:.1f}°)")
                score -= 25
                recommendations.append("Keep your ears aligned with the tops of your shoulders; avoid looking down.")

            if is_slouched_stand or is_forward_head_stand:
                pattern = "Standing: Slouched / Forward Lean"
            elif sh_tilt is not None and sh_tilt > 7.0:
                pattern = "Standing: Asymmetric / Uneven Stance"
            elif len(flags) > 0:
                pattern = "Standing: Sub-Optimal Stance"
            else:
                pattern = "Standing: Ideal Neutral Posture"

        # General healthy ergonomics advice if score is good
        if not recommendations:
            recommendations.append("Great posture! Maintain this neutral spine alignment.")
            recommendations.append("Remember the 20-20-20 rule: take short standing/movement breaks every 30-45 minutes.")

        # Final score clamping and status level
        score = max(10, min(100, score))
        if score >= 80:
            status_level = "GOOD"
        elif score >= 60:
            status_level = "WARNING"
        else:
            status_level = "ALERT"

        return pattern, status_level, score, flags, recommendations

    def process_image(
        self,
        image_bgr: np.ndarray,
        draw_annotations: bool = True,
        draw_angles: bool = True,
        draw_bbox: bool = True,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Runs YOLO11-Pose inference on an image and returns:
        - annotated_image (BGR)
        - list of analysis dicts for all detected people
        """
        results = self.model(image_bgr, conf=self.conf_threshold, verbose=False)
        annotated_image = image_bgr.copy() if draw_annotations else image_bgr
        people_analyses = []

        if len(results) == 0 or results[0].keypoints is None:
            return annotated_image, people_analyses

        res = results[0]
        if res.keypoints.xy is None or len(res.keypoints.xy) == 0:
            return annotated_image, people_analyses

        all_kpts = res.keypoints.xy.cpu().numpy()  # (N, 17, 2)
        all_confs = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(all_kpts.shape[:2])
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else [None] * len(all_kpts)

        for i in range(len(all_kpts)):
            kpts = all_kpts[i]
            confs = all_confs[i]
            box = boxes[i].tolist() if boxes[i] is not None else None

            # Skip detections with too few valid keypoints
            valid_kpts = np.sum(confs >= self.kpt_threshold)
            if valid_kpts < 4:
                continue

            analysis = self.analyze_person(kpts, confs, box)
            people_analyses.append(analysis)

            if draw_annotations:
                self._draw_posture_overlay(
                    annotated_image,
                    analysis,
                    draw_angles=draw_angles,
                    draw_bbox=draw_bbox,
                    person_index=i + 1,
                )

        return annotated_image, people_analyses

    def _draw_posture_overlay(
        self,
        img: np.ndarray,
        analysis: Dict[str, Any],
        draw_angles: bool = True,
        draw_bbox: bool = True,
        person_index: int = 1,
    ):
        """Draws skeleton, angle labels, bounding box, and posture badges."""
        kpts = analysis['keypoints']
        confs = analysis['confs']
        status = analysis['status_level']
        metrics = analysis['metrics']

        # Theme color based on status
        if status == "GOOD":
            main_color = (46, 204, 113)    # Bright Green (BGR)
            badge_color = (39, 174, 96)
        elif status == "WARNING":
            main_color = (0, 165, 255)    # Orange (BGR)
            badge_color = (0, 140, 255)
        else:
            main_color = (50, 50, 235)     # Vibrant Red (BGR)
            badge_color = (30, 30, 200)

        # 1. Draw Bounding Box if available
        bbox = analysis.get('bbox')
        if draw_bbox and bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), main_color, 2)
            # Person tag
            tag = f"Person #{person_index}: {analysis['score']}%"
            cv2.putText(img, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 2. Draw Skeleton Connections
        for pt1_idx, pt2_idx in SKELETON_CONNECTIONS:
            if confs[pt1_idx] >= self.kpt_threshold and confs[pt2_idx] >= self.kpt_threshold:
                p1 = (int(kpts[pt1_idx][0]), int(kpts[pt1_idx][1]))
                p2 = (int(kpts[pt2_idx][0]), int(kpts[pt2_idx][1]))
                cv2.line(img, p1, p2, main_color, 3, cv2.LINE_AA)

        # 3. Draw Keypoints
        for idx in range(len(kpts)):
            if confs[idx] >= self.kpt_threshold:
                pt = (int(kpts[idx][0]), int(kpts[idx][1]))
                cv2.circle(img, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, pt, 3, main_color, -1, cv2.LINE_AA)

        # 4. Draw Angle annotations & plumb lines
        if draw_angles:
            pts = analysis['points_used']
            sh_pt = (int(pts['shoulder'][0]), int(pts['shoulder'][1]))
            head_pt = (int(pts['head'][0]), int(pts['head'][1]))
            hip_pt = (int(pts['hip'][0]), int(pts['hip'][1]))
            knee_pt = (int(pts['knee'][0]), int(pts['knee'][1]))

            h, w = img.shape[:2]
            scale = max(0.5, min(1.2, h / 800.0))
            font_scale = 0.55 * scale
            font_thick = max(1, int(2 * scale))

            # Neck plumb line (vertical reference from shoulder)
            if analysis['visibility'].get('neck') and metrics.get('neck_angle') is not None:
                neck_val = metrics['neck_angle']
                cv2.line(img, sh_pt, (sh_pt[0], head_pt[1]), (220, 220, 220), 1, cv2.LINE_AA)
                label_pos = (sh_pt[0] + int(12 * scale), (sh_pt[1] + head_pt[1]) // 2)
                cv2.putText(img, f"Neck: {neck_val} deg", label_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thick + 1, cv2.LINE_AA)
                cv2.putText(img, f"Neck: {neck_val} deg", label_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

            # Torso angle label
            if analysis['visibility'].get('torso') and metrics.get('torso_angle') is not None:
                torso_val = metrics['torso_angle']
                label_pos = (hip_pt[0] + int(12 * scale), (sh_pt[1] + hip_pt[1]) // 2)
                cv2.putText(img, f"Torso: {torso_val} deg", label_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thick + 1, cv2.LINE_AA)
                cv2.putText(img, f"Torso: {torso_val} deg", label_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

            # Hip angle label
            if analysis['visibility'].get('hip') and metrics.get('hip_angle') is not None:
                hip_val = metrics['hip_angle']
                label_pos = (hip_pt[0] + int(12 * scale), hip_pt[1] - int(10 * scale))
                cv2.putText(img, f"Hip: {hip_val} deg", label_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thick + 1, cv2.LINE_AA)
                cv2.putText(img, f"Hip: {hip_val} deg", label_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

            # Knee angle label
            if analysis['visibility'].get('knee') and metrics.get('knee_angle') is not None:
                knee_val = metrics['knee_angle']
                label_pos = (knee_pt[0] + int(12 * scale), knee_pt[1])
                cv2.putText(img, f"Knee: {knee_val} deg", label_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thick + 1, cv2.LINE_AA)
                cv2.putText(img, f"Knee: {knee_val} deg", label_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

        # 5. Diagnostic Banner at Top
        h, w = img.shape[:2]
        scale = max(0.5, min(1.2, h / 800.0))
        banner_h = int(32 * scale)
        banner_y = int(banner_h * 1.2 * person_index)
        if banner_y < h - 40:
            badge_text = f"#{person_index} {analysis['pattern']} | Score: {analysis['score']}% [{status}]"
            font_size = 0.55 * scale
            font_thick = max(1, int(2 * scale))
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thick)
            pad = int(8 * scale)
            cv2.rectangle(img, (15, banner_y - th - pad), (15 + tw + 2 * pad, banner_y + pad), (20, 20, 20), -1)
            cv2.rectangle(img, (15, banner_y - th - pad), (15 + tw + 2 * pad, banner_y + pad), badge_color, max(1, int(2 * scale)))
            cv2.putText(img, badge_text, (15 + pad, banner_y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thick, cv2.LINE_AA)
