# file: models/loader.py

from object_detector import ObjectDetector
from caption_generator import CaptionGenerator
from semantic_matcher import SemanticMatcher

class ModelManager:
    """A class to load and hold all ML models, ensuring they are loaded only once."""
    def __init__(self):
        self.detector = None
        self.captioner = None
        self.matcher = None

    def load_all(self):
        """Loads all models into the instance attributes."""
        print("--- Loading all models into memory... ---")
        self.detector = ObjectDetector()
        self.captioner = CaptionGenerator(peft_model_path="./blip-finetuned-model")
        self.matcher = SemanticMatcher()
        print("--- ✅ All models loaded successfully. API is ready. ---")

# Create a single, global instance of the model manager that the app will use
models = ModelManager()