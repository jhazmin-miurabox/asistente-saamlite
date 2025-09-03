from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ruta de tu archivo
ruta_archivo = "tutoriales.txt"

# 1. Cargar el archivo
loader = TextLoader(ruta_archivo, encoding="utf-8")
documentos = loader.load()

# 2. Fragmentar el texto en chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
docs = splitter.split_documents(documentos)

# 3. Crear embeddings y el índice
embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
vector_db = FAISS.from_documents(docs, embeddings)

# 4. Guardar el índice en la carpeta (sobrescribe si ya existe)
vector_db.save_local("vectores/faiss_index")
