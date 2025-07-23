# run_caption.py

import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
import os

# --- Configuration ---
# 1. Path to your fine-tuned adapter model (the folder you downloaded)
PEFT_MODEL_PATH = "./blip-finetuned-model" 
# 2. Path to the image you want to caption
IMAGE_PATH = "Acura_ILX.jpg" # <--- CHANGE THIS
# 3. Original base model ID
BASE_MODEL_ID = "Salesforce/blip-image-captioning-base"
# ---------------------

# Check if a CUDA-enabled GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")


# Load the processor from your saved folder
processor = AutoProcessor.from_pretrained(PEFT_MODEL_PATH)

# Load the base model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = BlipForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    use_safetensors=True  # <-- Add this line
)
# Apply the LoRA adapter to the base model
model = PeftModel.from_pretrained(base_model, PEFT_MODEL_PATH)
model.eval() # Set the model to evaluation mode

print("✅ Model and processor loaded successfully.")

# --- Generate a caption ---
def generate_caption(image_path):
    """Loads an image and generates a caption using the fine-tuned model."""
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return

    try:
        # Load and process the image
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

        # Generate the caption
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        print("\n--- Caption ---")
        print(f"🤖 Generated: {generated_text}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the generation
if __name__ == "__main__":
    generate_caption(IMAGE_PATH)