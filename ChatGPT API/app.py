from flask import Flask, render_template, request, jsonify
import openai
from openai import OpenAI

client = OpenAI(api_key='')

# Configura tu clave API de OpenAI

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api", methods=["POST"])
def api():
    message = request.json.get("message")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Solicitud a la API de OpenAI
        completion = client.chat.completions.create(model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un Asistente financiero dirigido al público colombiano"},
            {"role": "user", "content": message}
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)

        response_content = completion.choices[0].message.content
        return jsonify({"response": response_content})

    except openai.OpenAIError as e:
        # Manejo específico para errores de la API de OpenAI
        print("Error de OpenAI:", str(e))
        return jsonify({"error": "OpenAI API error: " + str(e)}), 500
    except Exception as e:
        # Captura cualquier otro error general
        print("Error general:", str(e))
        return jsonify({"error": "Server error: " + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
