

from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import Dict, List
import requests

from models.loader import models

def _draw_detections(image: Image.Image, detections: List[Dict]) -> Image.Image:
    """Helper function to draw all bounding boxes on an image."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for det in detections:
        box = det['box']
        label = det['label']
        color = "blue" if label == 'car' else "green"
        draw.rectangle(box, outline=color, width=3)
        text_position = (box[0], box[1] - 25)
        draw.text(text_position, label, fill=color, font=font)
        
    return image

def run_pipeline(image_bytes: bytes, text_prompt: str, vehicle_parts_vocab: set) -> Dict:
    """
    Runs the full pipeline and returns the caption and Base64-encoded images.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 1. Detect objects using the model from the manager
        detected_objects = models.detector.detect_objects(image) 
        if not detected_objects:
            return {"error": "No objects were detected in the image."}

        annotated_image = _draw_detections(image.copy(), detected_objects)
        buffered_annotated = io.BytesIO()
        annotated_image.save(buffered_annotated, format="JPEG")
        annotated_image_base64 = base64.b64encode(buffered_annotated.getvalue()).decode("utf-8")

        # 2. Find the best match
        keywords = models.matcher.extract_keywords(text_prompt, vehicle_parts_vocab) 
        
        #best_match_object = models.matcher.find_best_match(keywords, detected_objects)
        
        #print(f"Best match object: {best_match_object}")
        
        response = requests.post(
            "https://374b27a6e238.ngrok-free.app/match/",
            json={"keywords": keywords, "detected_objects": detected_objects},
            timeout=10
        )
        if response.status_code != 200:
            return {"error": f"Match API failed: {response.status_code}"}
        
        print(f"Match API Response: {response.json()}")
        
        best_match_object = response.json()

        if not best_match_object:
            return {"error": "Could not find a confident match for the prompt."}

        if not best_match_object:
            return {"error": "Could not find a confident match for the prompt."}

        # 3. Crop the matched object
        cropped_image = image.crop(best_match_object['box'])

        # 4. Generate the caption
        final_caption = models.captioner.generate(cropped_image) 

        buffered_cropped = io.BytesIO()
        cropped_image.save(buffered_cropped, format="JPEG")
        cropped_image_base64 = base64.b64encode(buffered_cropped.getvalue()).decode("utf-8")

        return {
            "matched_object": best_match_object['label'],
            "caption": final_caption,
            "annotated_image_base64": annotated_image_base64,
            "cropped_image_base64": cropped_image_base64
        }

    except Exception as e:
        print(f"Pipeline Error: {e}")
        return {"error": "An internal error occurred during processing."}