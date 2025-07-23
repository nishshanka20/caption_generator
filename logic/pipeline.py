# file: logic/pipeline.py

from PIL import Image
import io
from typing import Dict

# Import the single instance of the model manager
from models.loader import models

def run_pipeline(image_bytes: bytes, text_prompt: str, vehicle_parts_vocab: set) -> Dict:
    """
    Runs the full pipeline and returns the caption and matched object name.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 1. Detect all objects using the model from the manager
        detected_objects = models.detector.detect_objects(image)
        if not detected_objects:
            return {"error": "No objects were detected in the image."}

        # 2. Find the best match
        keywords = models.matcher.extract_keywords(text_prompt, vehicle_parts_vocab)
        best_match_object = models.matcher.find_best_match(keywords, detected_objects)

        if not best_match_object:
            return {"error": "Could not find a confident match for the prompt."}

        # 3. Crop the matched object
        cropped_image = image.crop(best_match_object['box'])

        # 4. Generate the caption
        final_caption = models.captioner.generate(cropped_image)

        # 5. Return only the text data
        return {
            "matched_object": best_match_object['label'],
            "caption": final_caption,
        }

    except Exception as e:
        print(f"Pipeline Error: {e}")
        return {"error": "An internal error occurred during processing."}