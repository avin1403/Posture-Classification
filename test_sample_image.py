import os
import shutil
import cv2
from posture_analyzer import PostureAnalyzer

os.makedirs('samples', exist_ok=True)
shutil.copy(r'C:\python\Lib\site-packages\ultralytics\assets\bus.jpg', 'samples/bus.jpg')
shutil.copy(r'C:\python\Lib\site-packages\ultralytics\assets\zidane.jpg', 'samples/zidane.jpg')

analyzer = PostureAnalyzer(model_name="yolo11n-pose.pt")
img = cv2.imread('samples/bus.jpg')
annotated_img, results = analyzer.process_image(img)

print(f"Detected {len(results)} people in bus.jpg:")
for i, r in enumerate(results, 1):
    print(f"  Person #{i}:")
    print(f"    State: {r['state']} (view: {r['profile_view']})")
    print(f"    Pattern: {r['pattern']}")
    print(f"    Ergonomic Score: {r['score']}% [{r['status_level']}]")
    print(f"    Metrics: {r['metrics']}")
    print(f"    Flags: {r['flags']}")
    print(f"    Recommendations: {r['recommendations'][:2]}")

cv2.imwrite('samples/annotated_bus.jpg', annotated_img)
print("[PASS] Successfully processed bus.jpg and saved annotated_bus.jpg")
