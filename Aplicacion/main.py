from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
import openai

# Configuración de la API de OpenAI
openai.api_key = "tu_openai_key_aqui"  # Coloca aquí tu API Key

# Inicializar la aplicación FastAPI
app = FastAPI(title="Financial Bot API")

# Define la clase de datos para las preguntas
class FinancialQuery(BaseModel):
    question: str

# Crear el router para la versión de la API
api_router = APIRouter(prefix="/api/v1")

@api_router.post("/ask", tags=["Consultas Financieras"])
async def ask_question(query: FinancialQuery):
    try:
        # Llamada a la API de OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Cambia el modelo si lo necesitas
            messages=[
                {"role": "system", "content": "Eres un bot financiero que proporciona ayuda y asesoramiento financiero."},
                {"role": "user", "content": query.question}
            ]
        )
        answer = response.choices[0].message["content"]
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al procesar la solicitud") from e

# Incluir el router en la aplicación principal
app.include_router(api_router)
