from fastapi import FastAPI

from routes import predict

app = FastAPI()

app.include_router(predict.router)
