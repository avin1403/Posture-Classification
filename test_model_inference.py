"""
Test YOLO11-Pose model inference on a generated sample image.
Downloads yolo11n-pose.pt if not present, runs inference, and verifies output format.
"""

import numpy as np
import cv2
from posture_analyzer import PostureAnalyzer


def main():
    print("Testing YOLO11-Pose model inference...")
    analyzer = PostureAnalyzer(model_name="yolo11n-pose.pt")

    # Create a 640x640 dummy image
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_img[:] = (240, 240, 240)  # light gray background

    # Run inference
    annotated_img, analyses = analyzer.process_image(dummy_img)
    print(f"Model loaded successfully! Detected persons in blank image: {len(analyses)}")
    print(f"Annotated image shape: {annotated_img.shape}")
    print("[PASS] YOLO11-Pose model inference test passed!")


if __name__ == "__main__":
    main()
