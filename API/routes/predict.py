from fastapi import APIRouter

router = APIRouter(
    prefix="/api/predict",
    tags=["predict"],
    responses={404: {"description": "The requested URI was not found"}},
)

