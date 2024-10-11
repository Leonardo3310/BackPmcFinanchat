from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import predicciones as pred
from typing import List
from io import BytesIO

app = FastAPI()

class Opinion(BaseModel):
    opinion: str

@app.get("/")
def index():
    return {"message": "Hello World"}

@app.post("/prediccion/")
def hacer_prediccion(opiniones: List[Opinion]):
    opiniones = [opinion.opinion for opinion in opiniones]
    respuesta = pred.clasificacion(opiniones)
    return respuesta


@app.post("/reentrenar/")
async def reentrenar_modelo(file: UploadFile = File(...)):
    if file.content_type != 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        raise HTTPException(status_code=400, detail="File format not supported")

    try:
        contents = await file.read()
        excel_data = BytesIO(contents)
        df_retrain = pred.unir_datos(excel_data)

        matrics = pred.reentrenar_modelo(pred.modelo, df_retrain)

        return matrics

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante el reentrenamiento: {str(e)}")