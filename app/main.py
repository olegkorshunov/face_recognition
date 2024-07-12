from io import BytesIO

import numpy as np
import uvicorn
from database import DataBase
from fastapi import FastAPI, File, UploadFile
from get_face_embedding import GetFaceEmbedding
from PIL import Image


def load_image(data) -> np.ndarray:
    return np.array(Image.open(BytesIO(data)))


get_embedding = GetFaceEmbedding()
db = DataBase()

app = FastAPI()


@app.post("/add_user")
async def add_user(image: UploadFile = File(...)):
    embedding = get_embedding(load_image(await image.read()))
    db.add(embedding)
    return 200


@app.post("/check_user")
async def check(image: UploadFile = File(...)):
    embedding = get_embedding(load_image(await image.read()))
    db.check(embedding)
