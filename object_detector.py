# file: object_detector.py

from ultralytics import YOLO
from PIL import Image
from typing import List, Dict

class ObjectDetector:
    def __init__(self, model_name='yolov8n.pt'):
        print(f"Loading object detection model: {model_name}...")
        self.model = YOLO(model_name)
        print("✅ Object detector loaded successfully.")

    def detect_objects(self, image: Image.Image) -> List[Dict]: # <-- Takes a PIL Image now
        """Detects all objects in a PIL Image and returns their details."""
        try:
            results = self.model.predict(image, verbose=False)
            
            detected_objects = []
            for box in results[0].boxes:
                detected_class = results[0].names[int(box.cls)].lower()
                detected_objects.append({
                    "box": box.xyxy[0].tolist(),
                    "label": detected_class,
                    "confidence": box.conf[0].item()
                })
            
            print(f"🔎 Found {len(detected_objects)} objects in the image.")
            return detected_objects
            
        except Exception as e:
            print(f"❌ An error occurred during detection: {e}")
            return []