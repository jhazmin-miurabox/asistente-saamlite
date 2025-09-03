from flask import Flask, request, jsonify, render_template
import pathlib
import os
from flask_cors import CORS
# LangChain
from flask.cli import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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

# Clasificador de tema, usa gpt-3.5-turbo (más rápido y barato)
classifier = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)


ALLOWED_TOPICS_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
Eres un filtro de seguridad para un asistente integrado en un CRM que gestiona pólizas de seguro y endosos.
Debes analizar la pregunta del usuario y decidir si corresponde a alguno de los temas permitidos.

✅ **Temas permitidos**
- Funcionamiento y uso de la plataforma SAAM Lite (o la plataforma del usuario)
- Gestión de pólizas de seguro, coberturas, endosos, siniestros, **recibos / notas de crédito** y cobranza
- Consultas sobre agentes de seguros, clientes, reportes (pólizas, renovaciones, recibos), facturación y renovaciones
- Temas de seguros en general (auto, vida, gastos médicos)

❌ **Temas NO permitidos**
Cualquier asunto que no encaje en la lista anterior (por ejemplo: cultura general, deportes, geografía, etc.).

**Instrucciones de salida**
- Devuelve **solo** la palabra `permitido` si la pregunta corresponde a un tema autorizado.
- Devuelve **solo** la palabra `rechazado` si la pregunta no está permitida.

### Ejemplos
Pregunta: "¿Cómo genero los recibos de una póliza con pago trimestral?"  
Respuesta esperada: permitido

Pregunta: "¿Cuál es la capital de Francia?"  
Respuesta esperada: rechazado

Pregunta: "Muéstrame el reporte de cobranza de marzo"  
Respuesta esperada: permitido

Pregunta del usuario:
{question}
"""
)

# --------------------------------------------------
# Flask
# --------------------------------------------------
app = Flask(__name__)
CORS(app) 

@app.route("/")
def index():
    return render_template("index.html")    # tu página simple de prueba

@app.route("/ayuda", methods=["POST"])
def ayuda():
    data = request.get_json(force=True, silent=True) or {}
    pregunta = data.get("question")

    if not pregunta:
        return jsonify({"response": "Por favor, proporciona una pregunta válida."}), 400

 
    # --- Filtro temático ---
    filtro_prompt = ALLOWED_TOPICS_PROMPT.format(question=pregunta)
    try:
        filtro_resp = classifier.predict(filtro_prompt).strip().lower()
        if filtro_resp != "permitido":
            return jsonify({"response": "Lo siento, solo puedo ayudarte con temas relacionados a la plataforma, pólizas de seguro, endosos o coberturas. ¿Te gustaría preguntar algo sobre estos temas?"}), 200
    except Exception as e:
        # Si el filtro falla, NO respondas la pregunta por seguridad
        return jsonify({"response": "Ocurrió un error en el filtro de temas. Intenta de nuevo."}), 500

    # --- Si sí es permitido, consulta el vectorstore ---
    try:
        result = qa_chain({"query": pregunta})
        return jsonify({"response": result["result"].strip()})
    except Exception as e:
        return jsonify({"response": f"Ocurrió un error al procesar la solicitud: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)

