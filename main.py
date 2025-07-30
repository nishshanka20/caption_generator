

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Dict
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware  # <-- 1. Import the middleware

# Import your project modules
from models.loader import models
from logic.pipeline import run_pipeline
from utils import load_vehicle_parts

app = FastAPI(
    title="Object-Specific Image Captioner API",
    description="Upload an image and a prompt to detect an object and generate a caption for it.",
    version="1.0.0"
)

# --- 2. Add the CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# --------------------------------

class CaptionResponse(BaseModel):
    matched_object: str
    caption: str
    annotated_image_base64: str
    cropped_image_base64: str

@app.on_event("startup")
async def startup_event():
    models.load_all()

VEHICLE_VOCAB = load_vehicle_parts("vehicle_parts_2.json")

@app.get("/", tags=["General"])
async def read_root():
    return {"message": "Welcome to the BLIP Object Captioning API!"}


@app.post("/caption/", response_model=CaptionResponse, tags=["Captioning"])
async def create_caption(
    image: UploadFile = File(..., description="The image file to process."),
    prompt: str = Form(..., description="A text prompt describing the object of interest.")
) -> Dict:
    image_bytes = await image.read()
    result = run_pipeline(
        image_bytes=image_bytes,
        text_prompt=prompt,
        vehicle_parts_vocab=VEHICLE_VOCAB
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result