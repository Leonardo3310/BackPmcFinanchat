from flask import Flask, render_template, request, jsonify
import openai
from openai import OpenAI
from flask_cors import CORS

client = OpenAI(api_key='')

# Configura tu clave API de OpenAI
app = Flask(__name__)
CORS(app)

# Historial de mensajes (contexto)
conversation_history = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api", methods=["POST"])
def api():
    global conversation_history  # Usamos una variable global para mantener el historial

    message = request.json.get("message")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Agregar el mensaje del usuario al historial
    conversation_history.append({"role": "user", "content": message})

    try:
        # Solicitud a la API de OpenAI con todo el historial
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "Eres un Asistente financiero (Te llamas Financhat) dirigido al público colombiano. Quiero que las respuestas sean detalladas, enfocadas en que las personas aprendan finanzas. Que nunca diga recomendaciones en particular, sino que se guíe al usuario a la mejor alternativa."}
            ] + conversation_history,
            temperature=1,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Respuesta del bot
        response_content = completion.choices[0].message.content

        # Validar y ajustar los delimitadores de LaTeX
        response_content = validate_latex_format(response_content)

        print(response_content)
        conversation_history.append({"role": "assistant", "content": response_content})

        return jsonify({"response": response_content})

    except openai.OpenAIError as e:
        # Manejo específico para errores de la API de OpenAI
        print("Error de OpenAI:", str(e))
        return jsonify({"error": "OpenAI API error: " + str(e)}), 500
    except Exception as e:
        # Captura cualquier otro error general
        print("Error general:", str(e))
        return jsonify({"error": "Server error: " + str(e)}), 500
    
def validate_latex_format(content):
    """
    Reemplaza delimitadores incorrectos en las fórmulas de LaTeX.
    """
    # Reemplazar corchetes por delimitadores correctos
    content = content.replace("[", "$$").replace("]", "$$")

    # Asegurarse de que las fórmulas en línea usen \( ... \)
    content = content.replace("\\[", "\\(").replace("\\]", "\\)")

    return content

if __name__ == '__main__':
    app.run(debug=True)