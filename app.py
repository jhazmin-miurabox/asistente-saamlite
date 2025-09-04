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

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Eres un asistente de seguros. Usa el contexto para responder de forma breve, "
        "en máximo tres oraciones, y mantén solo la información esencial. "
        "Incluye enlaces si aparecen en el contexto.\n\nContexto:\n{context}\n\nPregunta: {question}\n\nRespuesta:"
    ),
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,  # pon True si quieres devolver las fuentes
    chain_type_kwargs={"prompt": QA_PROMPT},
)

# Clasificador de tema, usa gpt-3.5-turbo (más rápido y barato)
classifier = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)


def es_saludo(texto: str) -> bool:
    """Detecta si el texto es un saludo simple."""
    if not texto:
        return False
    texto = texto.strip().lower()
    saludos = (
        "hola",
        "buenos dias",
        "buenos días",
        "buenas tardes",
        "buenas noches",
        "hi",
        "hello",
    )
    return any(texto.startswith(s) for s in saludos)


def es_tema_seguro(texto: str) -> bool:
    """Detecta palabras clave relacionadas con seguros."""
    if not texto:
        return False
    texto = texto.lower()
    palabras = (
        "seguro",
        "seguros",
        "póliza",
        "poliza",
        "endoso",
        "cobertura",
        "siniestro",
    )
    return any(p in texto for p in palabras)


def respuesta_crear_poliza(texto: str) -> str:
    """Devuelve una respuesta breve con enlace para crear pólizas."""
    if not texto:
        return ""
    t = texto.lower()
    if "crear" in t and ("póliza" in t or "poliza" in t):
        return (
            "Para crear una póliza ve al módulo de Pólizas y haz clic en 'Nueva Póliza'. "
            "Haz clic aquí para ir directo al formulario: /formulario-polizas"
        )
    return ""


def respuesta_crear_grupo(texto: str) -> str:
    """Devuelve una respuesta breve con enlace para crear grupos."""
    if not texto:
        return ""
    t = texto.lower()
    if "crear" in t and "grupo" in t:
        return (
            "Para crear un grupo ve al módulo de Grupos y haz clic en 'Nuevo Grupo'. "
            "Haz clic aquí para ir directo al formulario: /grupos/grupo"
        )
    return ""


def es_saludo(texto: str) -> bool:
    """Detecta si el texto es un saludo simple."""
    if not texto:
        return False
    texto = texto.strip().lower()
    saludos = (
        "hola",
        "buenos dias",
        "buenos días",
        "buenas tardes",
        "buenas noches",
        "hi",
        "hello",
    )
    return any(texto.startswith(s) for s in saludos)


def es_tema_seguro(texto: str) -> bool:
    """Detecta palabras clave relacionadas con seguros."""
    if not texto:
        return False
    texto = texto.lower()
    palabras = (
        "seguro",
        "sgiteguros",
        "póliza",
        "poliza",
        "endoso",
        "cobertura",
        "siniestro",
    )
    return any(p in texto for p in palabras)



def respuesta_crear_endoso(texto: str):
    """Proporciona una respuesta directa si el usuario pregunta cómo crear un endoso."""
    if not texto:
        return None

    pregunta = texto.lower()
    claves = (
        "crear endoso",
        "crear un endoso",
        "nuevo endoso",
        "generar endoso",
        "alta de endoso",
    )

    if any(c in pregunta for c in claves):
        return (
            "Para crear un endoso, primero abre la póliza vigente y en la sección Opciones selecciona "
            "\"Crear Endoso\". Luego, elige el tipo de endoso (A, B o D), selecciona el concepto, "
            "define la vigencia, completa el formulario correspondiente y, si aplica, genera los recibos o notas "
            "de crédito. Finalmente, revisa los datos y guarda el endoso para que se aplique en la póliza."
        )

    return None


def respuesta_crear_poliza(texto: str):
    """Proporciona una respuesta directa si el usuario pregunta cómo crear una póliza."""
    if not texto:
        return None

    pregunta = texto.lower()
    claves = (
        "crear poliza",
        "crear póliza",
        "nueva poliza",
        "nueva póliza",
        "alta de poliza",
        "alta de póliza",
    )

    if any(c in pregunta for c in claves):
        return (
            "Para crear una póliza:\n"
            "1. Ve al módulo Pólizas y elige 'Nueva Póliza'.\n"
            "2. Completa el formulario correspondiente.\n"
            "3. Guarda para finalizar. También puedes ir al formulario en "
            "http://pruebas.localhost:3000/formulario-polizas"
        )

    return None



def respuesta_crear_grupo(texto: str):
    """Responde con pasos para crear un grupo."""
    if not texto:
        return None

    pregunta = texto.lower()
    claves = (
        "crear grupo",
        "crear un grupo",
        "nuevo grupo",
        "alta de grupo",
    )

    if any(c in pregunta for c in claves):
        return (
            "Para crear un grupo:\n"
            "1. En el menú lateral, selecciona 'Grupos'.\n"
            "2. Haz clic en 'Nuevo Grupo' y completa nombre y descripción.\n"
            "3. Revisa la información y guarda para finalizar."
        )

    return None


def respuesta_crear_contratante_fisico(texto: str):
    """Responde cómo crear un contratante físico."""
    if not texto:
        return None

    pregunta = texto.lower()
    if "crear" not in pregunta:
        return None

    claves = (
        "contratante fisico",
        "contratante físico",
    )

    if any(c in pregunta for c in claves):
        return (
            "Para crear un contratante físico:\n"
            "1. Ve a Clientes > Contratantes y selecciona 'Nuevo Contratante'.\n"
            "2. Elige tipo Físico y captura nombre, apellidos, fecha de nacimiento, género y RFC.\n"
            "3. Añade correo, teléfono y direcciones si aplica, luego guarda."
        )

    return None


def respuesta_crear_contratante_moral(texto: str):
    """Responde cómo crear un contratante moral."""
    if not texto:
        return None

    pregunta = texto.lower()
    if "crear" not in pregunta:
        return None

    if "contratante" in pregunta and "moral" in pregunta:
        return (
            "Para crear un contratante moral:\n"
            "1. Ve a Clientes > Contratantes y elige 'Nuevo Contratante'.\n"
            "2. Selecciona tipo Moral y captura nombre de la empresa, fecha de constitución y RFC.\n"
            "3. Completa datos de contacto y direcciones, luego guarda."
        )

    return None


def respuesta_crear_recibos(texto: str):
    """Responde cómo generar recibos de póliza o endoso."""
    if not texto:
        return None

    pregunta = texto.lower()
    claves = (
        "crear recibo",
        "crear recibos",
        "generar recibo",
        "generar recibos",
    )

    if any(c in pregunta for c in claves):
        return (
            "Para generar recibos:\n"
            "1. Dentro del formulario de póliza o endoso, abre la sección 'Recibos'.\n"
            "2. Captura prima neta, descuentos, RPF, derecho e IVA.\n"
            "3. Selecciona la frecuencia de pago y usa 'Generar Primer Recibo' y luego 'Calcular y Generar Recibos'."
        )

    return None


def respuesta_donde_ver(texto: str):
    """Responde dónde ver distintos elementos en la plataforma."""
    if not texto:
        return None

    pregunta = texto.lower()
    if "donde ver" not in pregunta and "dónde ver" not in pregunta:
        return None

    if "poliza" in pregunta or "póliza" in pregunta:
        return "Puedes ver tus pólizas en el módulo Pólizas: http://pruebas.localhost:3000/polizas"
    if "grupo" in pregunta:
        return "Selecciona 'Grupos' en el menú lateral para verlos."
    if "contratante" in pregunta:
        return "Ve a Clientes > Contratantes en el menú para consultar la lista."
    if "recibo" in pregunta:
        return "Abre la póliza o endoso y entra a la sección 'Recibos' para verlos."

    return None


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

    if es_saludo(pregunta):
        return jsonify({"response": "¡Hola! ¿En qué puedo ayudarte?"}), 200

    direct = respuesta_crear_poliza(pregunta)
    if not direct:
        direct = respuesta_crear_grupo(pregunta)
    if direct:
        return jsonify({"response": direct}), 200

    permitido = es_tema_seguro(pregunta)

    if not permitido:
        # --- Filtro temático con LLM ---
        filtro_prompt = ALLOWED_TOPICS_PROMPT.format(question=pregunta)
        try:
            filtro_resp = classifier.invoke(filtro_prompt).content.strip().lower()
            permitido = filtro_resp.startswith("permitido")
        except Exception:
            return jsonify({"response": "Ocurrió un error en el filtro de temas. Intenta de nuevo."}), 500

    if not permitido:
        return jsonify({
            "response": (
                "Lo siento, solo puedo ayudarte con temas relacionados a la plataforma, pólizas de seguro, "
                "endosos o coberturas. ¿Te gustaría preguntar algo sobre estos temas?"
            )
        }), 200

    # --- Si sí es permitido, consulta el vectorstore ---
    try:
        result = qa_chain({"query": pregunta})
        respuesta = result.get("result", "").strip()
        if not respuesta:
            respuesta = llm.invoke(pregunta).content.strip()
        return jsonify({"response": respuesta})
    except Exception:
        try:
            # Fallback directo al modelo si falla el retriever
            return jsonify({"response": llm.invoke(pregunta).content.strip()})
        except Exception as inner_e:
            return jsonify({"response": f"Ocurrió un error al procesar la solicitud: {str(inner_e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)

