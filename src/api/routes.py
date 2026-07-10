import cv2
import numpy as np

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.prediction.predictor import Predictor

router = APIRouter()

predictor = Predictor()


@router.get("/health")
async def health():
    return {"status": "ok", "message": "Service is up and running"}


@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are allowed."
        )

    contents = await file.read()

    image = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    result = predictor.predict(image)

    return result