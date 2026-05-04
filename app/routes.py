import os
from fastapi import APIRouter, File, Request, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.model import load_model, predict

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", 5))
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Load model once at startup — will be None until weights are available
model = load_model()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@router.post("/predict")
async def predict_species(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Please upload a JPEG, PNG, or WebP image."
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {MAX_IMAGE_SIZE_MB}MB size limit."
        )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model weights not found. The model has not been trained yet."
        )

    result = predict(image_bytes, model)
    return result