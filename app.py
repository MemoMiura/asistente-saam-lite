from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import openai 
from langchain_openai import OpenAI

# Cargar variables de entorno
load_dotenv()

# Inicializar Flask
app = Flask(__name__)

# Configurar la clave de API de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Crear una instancia del modelo LLM de OpenAI usando LangChain
llm = OpenAI()

# Ruta para la interfaz de usuario
@app.route("/")
def index():
    return render_template("index.html")

# Ruta para procesar las peticiones
@app.route("/ayuda", methods=["POST"])
def ayuda():
    user_input = request.json.get("question")
    if user_input:
        response = llm(user_input)
        return jsonify({"response": response})
    return jsonify({"response": "Por favor, proporciona una pregunta."}), 400


if __name__ == "__main__":
    app.run(debug=True)
