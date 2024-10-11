from fastapi import FastAPI
from pydantic import BaseModel
import predicciones as pred
from typing import List

app = FastAPI()

class Opinion(BaseModel):
    opinion: str

@app.get("/")
def index():
    return {"message": "Hello World"}

@app.post("/prediccion/")
def hacer_prediccion(opiniones: List[Opinion]):
    respuesta = pred.clasificacion(opiniones)
    return respuesta
