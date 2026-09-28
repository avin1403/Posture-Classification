"""
Unit and integration tests for PostureAnalyzer.
Tests mathematical angle functions, sitting/standing pattern classifier,
and end-to-end YOLO11 model processing.
"""

import numpy as np
from posture_analyzer import PostureAnalyzer, KeypointIndices


def test_angle_3pt():
    """Verify 3-point angle calculation for 90 deg and 180 deg."""
    # Right angle: (0, 1) -> (0, 0) -> (1, 0)
    angle_90 = PostureAnalyzer.calculate_angle_3pt((0, 1), (0, 0), (1, 0))
    assert abs(angle_90 - 90.0) < 1e-4, f"Expected 90, got {angle_90}"

    # Straight line: (0, 1) -> (0, 0) -> (0, -1)
    angle_180 = PostureAnalyzer.calculate_angle_3pt((0, 1), (0, 0), (0, -1))
    assert abs(angle_180 - 180.0) < 1e-4, f"Expected 180, got {angle_180}"
    print("[PASS] 3-point angle tests passed.")


def test_vertical_angle():
    """Verify angle relative to downward plumb line."""
    # Perfectly vertical: head (100, 50), shoulder (100, 150)
    vert_angle = PostureAnalyzer.calculate_vertical_angle((100, 50), (100, 150))
    assert abs(vert_angle - 0.0) < 1e-4, f"Expected 0, got {vert_angle}"

    # 45-degree tilt
    tilt_45 = PostureAnalyzer.calculate_vertical_angle((150, 50), (100, 100))
    assert abs(tilt_45 - 45.0) < 1e-4, f"Expected 45, got {tilt_45}"
    print("[PASS] Vertical angle tests passed.")


def test_sitting_posture_classification():
    """Test synthetic keypoints for good sitting vs slouching."""
    analyzer = PostureAnalyzer()

    # Synthetic good sitting posture:
    # head over shoulder, torso vertical, knees at 90 deg, hip at 90 deg
    kpts_good = np.zeros((17, 2), dtype=np.float32)
    confs_good = np.zeros(17, dtype=np.float32)

    # Left profile
    left_idxs = [KeypointIndices.LEFT_EAR, KeypointIndices.LEFT_SHOULDER, KeypointIndices.LEFT_HIP, KeypointIndices.LEFT_KNEE, KeypointIndices.LEFT_ANKLE]
    for idx in left_idxs:
        confs_good[idx] = 1.0
    kpts_good[KeypointIndices.LEFT_EAR] = [200, 100]
    kpts_good[KeypointIndices.LEFT_SHOULDER] = [200, 180]
    kpts_good[KeypointIndices.LEFT_HIP] = [200, 320]
    kpts_good[KeypointIndices.LEFT_KNEE] = [280, 320]   # horizontal thigh
    kpts_good[KeypointIndices.LEFT_ANKLE] = [280, 420]  # vertical shin

    analysis_good = analyzer.analyze_person(kpts_good, confs_good)
    assert analysis_good['state'] == 'Sitting', f"Expected Sitting, got {analysis_good['state']}"
    assert "Ideal" in analysis_good['pattern'], f"Expected Ideal in pattern, got {analysis_good['pattern']}"
    assert analysis_good['score'] >= 90
    print(f"[PASS] Good sitting classified: {analysis_good['pattern']} (Score: {analysis_good['score']}%)")

    # Synthetic slouched sitting:
    # Torso leaning forward: shoulder (260, 200), hip (200, 320) -> tilt ~26 deg
    # Head craning forward: ear (310, 120)
    kpts_slouch = np.zeros((17, 2), dtype=np.float32)
    confs_slouch = np.zeros(17, dtype=np.float32)
    for idx in left_idxs:
        confs_slouch[idx] = 1.0

    kpts_slouch[KeypointIndices.LEFT_EAR] = [310, 120]
    kpts_slouch[KeypointIndices.LEFT_SHOULDER] = [260, 200]
    kpts_slouch[KeypointIndices.LEFT_HIP] = [200, 320]
    kpts_slouch[KeypointIndices.LEFT_KNEE] = [280, 320]
    kpts_slouch[KeypointIndices.LEFT_ANKLE] = [280, 420]

    analysis_slouch = analyzer.analyze_person(kpts_slouch, confs_slouch)
    assert analysis_slouch['state'] == 'Sitting'
    assert "Slouch" in analysis_slouch['pattern'] or "Forward Head" in analysis_slouch['pattern']
    assert analysis_slouch['score'] < 70
    print(f"[PASS] Slouched sitting classified: {analysis_slouch['pattern']} (Score: {analysis_slouch['score']}%)")


def test_standing_posture_classification():
    """Test synthetic keypoints for good standing."""
    analyzer = PostureAnalyzer()

    # Standing vertically aligned
    kpts_stand = np.zeros((17, 2), dtype=np.float32)
    confs_stand = np.zeros(17, dtype=np.float32)
    left_idxs = [KeypointIndices.LEFT_EAR, KeypointIndices.LEFT_SHOULDER, KeypointIndices.LEFT_HIP, KeypointIndices.LEFT_KNEE, KeypointIndices.LEFT_ANKLE]
    for idx in left_idxs:
        confs_stand[idx] = 1.0

    kpts_stand[KeypointIndices.LEFT_EAR] = [200, 100]
    kpts_stand[KeypointIndices.LEFT_SHOULDER] = [200, 160]
    kpts_stand[KeypointIndices.LEFT_HIP] = [200, 280]
    kpts_stand[KeypointIndices.LEFT_KNEE] = [200, 400]
    kpts_stand[KeypointIndices.LEFT_ANKLE] = [200, 520]

    analysis_stand = analyzer.analyze_person(kpts_stand, confs_stand)
    assert analysis_stand['state'] == 'Standing', f"Expected Standing, got {analysis_stand['state']}"
    assert "Ideal" in analysis_stand['pattern']
    assert analysis_stand['score'] >= 85
    print(f"[PASS] Good standing classified: {analysis_stand['pattern']} (Score: {analysis_stand['score']}%)")


if __name__ == "__main__":
    print("Running posture analyzer tests...")
    test_angle_3pt()
    test_vertical_angle()
    test_sitting_posture_classification()
    test_standing_posture_classification()
    print("\nAll unit tests passed successfully!")
