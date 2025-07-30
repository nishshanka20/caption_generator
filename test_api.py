# file: test_api.py

import requests
import base64
import io
from PIL import Image
import argparse
import os

def get_content_type(file_path):
    """Determines the image content type based on the file extension."""
    extension = os.path.splitext(file_path)[1].lower()
    if extension in [".jpg", ".jpeg"]:
        return "image/jpeg"
    elif extension == ".png":
        return "image/png"
    elif extension == ".bmp":
        return "image/bmp"
    # Add other types as needed
    else:
        return "application/octet-stream" # A generic fallback

def test_endpoint(api_url: str, image_path: str, prompt: str):
    """Sends a request to the API and processes the response."""
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found at '{image_path}'")
        return

    print(f"Sending request for image: {image_path} with prompt: '{prompt}'")

    with open(image_path, "rb") as image_file:
        # --- FIX: Dynamically get the content type ---
        content_type = get_content_type(image_path)
        files = {"image": (os.path.basename(image_path), image_file, content_type)}
        # ---------------------------------------------
        
        data = {"prompt": prompt}
        
        try:
            response = requests.post(api_url, files=files, data=data)
        except requests.exceptions.ConnectionError as e:
            print(f"\n❌ Connection Error: Could not connect to the API at {api_url}")
            print("   Please make sure your Uvicorn server is running.")
            return

    # --- Process the response ---
    if response.status_code == 200:
        data = response.json()
        
        print("\n✅ Success! API Response:")
        print(f"   - Matched Object: {data['matched_object']}")
        print(f"   - Generated Caption: {data['caption']}")

        # Decode and show the ANNOTATED image
        annotated_base64 = data['annotated_image_base64']
        annotated_bytes = base64.b64decode(annotated_base64)
        annotated_image = Image.open(io.BytesIO(annotated_bytes))
        print("\n🖼️  Displaying the image with ALL detections...")
        annotated_image.show(title="All Detections")

        # Decode and show the CROPPED image
        cropped_base64 = data['cropped_image_base64']
        cropped_bytes = base64.b64decode(cropped_base64)
        cropped_image = Image.open(io.BytesIO(cropped_bytes))
        print("\n🖼️  Displaying the CROPPED image...")
        cropped_image.show(title="Cropped Image")

    else:
        print(f"\n❌ Error: {response.status_code}")
        print(f"   - Detail: {response.json().get('detail', 'No details provided.')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Object Captioning API.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file (e.g., .jpg, .png).")
    parser.add_argument("--prompt", type=str, required=True, help="The object to detect and caption.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/caption/", help="URL of the API endpoint.")
    
    args = parser.parse_args()
    
    test_endpoint(api_url=args.url, image_path=args.image, prompt=args.prompt)