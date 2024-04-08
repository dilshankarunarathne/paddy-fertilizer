from fastapi import APIRouter

router = APIRouter(
    prefix="/api/predict",
    tags=["predict"],
    responses={404: {"description": "The requested URI was not found"}},
)


@router.post("/")
async def predict_image_class(
        image: UploadFile = File(...)
):
    """
    The endpoint for predicting the class of an image

    Args:
        image_data (UploadFile): the image to predict

    Returns:
        (str) The class of the image
    """
    return {"class": "Class", "confidence": 0.0}
