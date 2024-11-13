from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os


openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class FinancialQuery(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Financial bot API with ChatGPT"}

@app.post("/ask")
async def ask_question(query: FinancialQuery):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Ajusta el modelo según tu suscripción
            messages=[
                {"role": "system", "content": "Eres un bot financiero que proporciona ayuda y asesoramiento financiero."},
                {"role": "user", "content": query.question}
            ]
        )
        answer = response.choices[0].message["content"]
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al procesar la solicitud") from e
