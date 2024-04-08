from fastapi import APIRouter, UploadFile, File
from PIL import Image
import numpy as np
from io import BytesIO
from lcc.main import predict_image_class, estimate_npa

router = APIRouter(
    prefix="/api/predict",
    tags=["predict"],
    responses={404: {"description": "The requested URI was not found"}},
)


@router.post("/")
async def predict_image(image: UploadFile = File(...)):
    """
    The endpoint for predicting the class of an image

    Args:
        image (UploadFile): the image to predict

    Returns:
        (str) The class of the image
        (str) The confidence score
        (str) The estimated Nitrogen requirement per Acre
    """
    image_data = await image.read()

    image = Image.open(BytesIO(image_data))
    image_array = np.array(image)

    class_name, confidence_score = predict_image_class(image_array)

    class_name = class_name.strip()

    confidence_score = float(confidence_score)

    npa_estimate = estimate_npa(class_name)

    return {"lcc": class_name, "confidence_score": confidence_score, "npa_estimate": npa_estimate}
