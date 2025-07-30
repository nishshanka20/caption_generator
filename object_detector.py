# file: object_detector.py (Updated for the Two-Model Pipeline)

from ultralytics import YOLO
from PIL import Image
from typing import List, Dict

class ObjectDetector:
    def __init__(self, general_model_name: str, parts_model_path: str):
        """
        Initializes the detector by loading TWO models:
        1. A general model to find whole vehicles.
        2. fine-tuned model to find specific parts.
        """
        print(f"Loading general detection model: {general_model_name}...")
        self.general_model = YOLO(general_model_name)
        print("✅ General model loaded.")

        print(f"Loading fine-tuned parts model: {parts_model_path}...")
        self.parts_model = YOLO(parts_model_path)
        print("✅ Fine-tuned parts model loaded.")

    def _extract_detections(self, results) -> List[Dict]:
        """Helper function to extract object data from a YOLO results object."""
        detected_objects = []
        if results[0].boxes:
            for box in results[0].boxes:
                detected_class = results[0].names[int(box.cls)].lower()
                detected_objects.append({
                    "box": box.xyxy[0].tolist(),
                    "label": detected_class,
                    "confidence": box.conf[0].item()
                })
        return detected_objects

    def detect_objects(self, image: Image.Image) -> List[Dict]:
        """
        Detects objects using both models and combines the results.
        """
        all_detections = []

        print("🔎 Detecting general objects (like 'car')...")
        general_results = self.general_model.predict(image, classes=[2], verbose=False)
        general_detections = self._extract_detections(general_results)
        all_detections.extend(general_detections)
        
        # 2. Run detection with fine-tuned parts model
        print("🔎 Detecting specific car parts...")
        parts_results = self.parts_model.predict(image, verbose=False)
        parts_detections = self._extract_detections(parts_results)
        all_detections.extend(parts_detections)

        print(f"✅ Found {len(all_detections)} total objects (car + parts).")
        print("\n--- All Detected Objects ---")
        if all_detections:
            for det in all_detections:
                print(f"   - Label: {det['label']}, Confidence: {det['confidence']:.2f}")
        else:
            print("   - No objects were detected.")
        print("--------------------------")
        return all_detections