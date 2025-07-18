from flask import Flask, request, jsonify, render_template
import pathlib
import os
import logging

# LangChain
from flask.cli import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Cargar variables de entorno desde .env
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "La variable de entorno OPENAI_API_KEY no está configurada."
    )

INDEX_DIR = pathlib.Path("faiss_index")  # carpeta donde guardaste el índice

if not INDEX_DIR.exists():
    raise FileNotFoundError(
        f"No se encontró el índice FAISS en {INDEX_DIR}. "
        "Ejecuta primero la celda que construye el índice."
    )

# --------------------------------------------------
# Inicializar modelos y retriever
# --------------------------------------------------
try:
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_db = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,  # necesario al cargar localmente
    )
except Exception as exc:
    logger.exception("Error al inicializar los modelos o cargar el índice")
    raise RuntimeError("Falló la inicialización del asistente") from exc

retriever = vector_db.as_retriever(search_kwargs={"k": 4})  # top-4 trozos relevantes
llm       = ChatOpenAI(temperature=0, model="gpt-4.1-mini", api_key=OPENAI_API_KEY)

# Instrucciones para restringir el dominio de respuestas
system_msg = (
    "Eres un asistente virtual especializado en un CRM para la "
    "gestión de pólizas de seguros. Responde exclusivamente preguntas "
    "relacionadas con dicho CRM y la administración de pólizas. "
    "Si la pregunta no está relacionada, indica que solo puedes ayudar "
    "con consultas sobre pólizas de seguros."
)
system_template = (
    f"{system_msg}\n"
    "Utiliza el siguiente contexto para responder de forma concisa. "
    "Si no conoces la respuesta basada en el contexto, responde que no lo sabes."
    "\n----------------\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=False   # pon True si quieres devolver las fuentes
)

# --------------------------------------------------
# Flask
# --------------------------------------------------
app = Flask(__name__)


@app.errorhandler(Exception)
def handle_exception(err):
    """Manejador global de errores para respuestas JSON consistentes."""
    logger.exception("Excepción no controlada: %s", err)
    return jsonify({"response": "Ocurrió un error interno."}), 500

@app.route("/")
def index():
    return render_template("index.html")    # tu página simple de prueba


@app.route("/salud")
def health():
    """Punto sencillo de verificación de estado."""
    return jsonify({"status": "ok"})

@app.route("/ayuda", methods=["POST"])
def ayuda():
    data = request.get_json(force=True, silent=True) or {}
    pregunta = data.get("question")

    if not pregunta:
        return jsonify({"response": "Por favor, proporciona una pregunta válida."}), 400

    try:
        result = qa_chain({"query": pregunta})
        logger.info("Pregunta procesada correctamente")
        return jsonify({"response": result["result"].strip()})
    except Exception as e:
        logger.exception("Error al procesar la solicitud")
        return jsonify({"response": f"Ocurrió un error al procesar la solicitud: {str(e)}"}), 500

if __name__ == "__main__":
    # Usa host y puerto según tu despliegue; debug=True solo en desarrollo
    debug_mode = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
