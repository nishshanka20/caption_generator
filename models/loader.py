# file: models/loader.py (Updated to initialize the two-model detector)

from object_detector import ObjectDetector
from caption_generator import CaptionGenerator
from semantic_matcher import SemanticMatcher
import os

class ModelManager:
    """A class to load and hold all ML models."""
    def __init__(self):
        self.detector = None
        self.captioner = None
        self.matcher = None

    def load_all(self):
        """Loads all models into the instance attributes."""
        global detector, captioner, matcher
        print("--- Loading all models into memory... ---")
        
        # Place your 'best.pt' file in the main project folder.
        fine_tuned_yolo_path = "best.pt" 
        if not os.path.exists(fine_tuned_yolo_path):
            raise FileNotFoundError(f"Could not find the fine-tuned model at '{fine_tuned_yolo_path}'. Please place 'best.pt' in the main project directory.")

        # Initialize the ObjectDetector with paths to both models
        self.detector = ObjectDetector(
            general_model_name='yolov8n.pt',
            parts_model_path=fine_tuned_yolo_path
        )
        
        self.captioner = CaptionGenerator(peft_model_path="./blip-finetuned-model")
        self.matcher = SemanticMatcher()
        print("--- ✅ All models loaded successfully. API is ready. ---")

# Create a single, global instance of the model manager
models = ModelManager()