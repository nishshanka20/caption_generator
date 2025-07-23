# file: caption_generator.py

import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

class CaptionGenerator:
    """
    A class to handle caption generation using a fine-tuned BLIP model.
    """
    def __init__(self, peft_model_path: str, base_model_id: str = "Salesforce/blip-image-captioning-base"):
        """
        Initializes the captioner by loading the base model and applying the PEFT adapter.
        """
        print(f"Loading captioning model from adapter: {peft_model_path}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the processor from the saved fine-tuned model directory
        self.processor = AutoProcessor.from_pretrained(peft_model_path)

        # Configure and load the 4-bit quantized base model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = BlipForConditionalGeneration.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map={"": 0},
            use_safetensors=True
        )

        # Apply the LoRA adapter to the base model
        self.model = PeftModel.from_pretrained(base_model, peft_model_path)
        self.model.eval()
        print("✅ Caption generator loaded successfully.")

    def generate(self, image_object: Image.Image) -> str:
        """
        Generates a caption for a given PIL Image object.

        Args:
            image_object (PIL.Image.Image): The image to be captioned.

        Returns:
            str: The generated caption text.
        """
        print("✍️  Generating caption for the cropped image...")
        inputs = self.processor(images=image_object, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=50)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        return generated_text