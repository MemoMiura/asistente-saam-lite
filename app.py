from flask import Flask, request, jsonify, render_template
import pathlib
import os

# LangChain
from flask.cli import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Cargar variables de entorno desde .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INDEX_DIR      = pathlib.Path("vectores/faiss_index")  # carpeta donde guardaste el índice

if not INDEX_DIR.exists():
    raise FileNotFoundError(
        f"No se encontró el índice FAISS en {INDEX_DIR}. "
        "Ejecuta primero la celda que construye el índice."
    )

# --------------------------------------------------
# Inicializar modelos y retriever
# --------------------------------------------------
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vector_db  = FAISS.load_local(
    str(INDEX_DIR),
    embeddings,
    allow_dangerous_deserialization=True,   # necesario al cargar localmente
)

retriever = vector_db.as_retriever(search_kwargs={"k": 4})  # top-4 trozos relevantes
llm       = ChatOpenAI(temperature=0, model="gpt-4.1-mini", api_key=OPENAI_API_KEY)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False   # pon True si quieres devolver las fuentes
)

# --------------------------------------------------
# Flask
# --------------------------------------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")    # tu página simple de prueba

@app.route("/ayuda", methods=["POST"])
def ayuda():
    data = request.get_json(force=True, silent=True) or {}
    pregunta = data.get("question")

    if not pregunta:
        return jsonify({"response": "Por favor, proporciona una pregunta válida."}), 400

    try:
        result = qa_chain({"query": pregunta})
        return jsonify({"response": result["result"].strip()})
    except Exception as e:
        return jsonify({"response": f"Ocurrió un error al procesar la solicitud: {str(e)}"}), 500

if __name__ == "__main__":
    # Usa host y puerto según tu despliegue; debug=True solo en desarrollo
    app.run(debug=True)
