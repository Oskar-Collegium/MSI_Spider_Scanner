from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Spider Scanner", version="1.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")

from app.routes import router
app.include_router(router)