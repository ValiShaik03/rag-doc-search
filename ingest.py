import os, json
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DATA_DIR = BASE_DIR / "data"
CONFIG_PATH = DATA_DIR / "embedding_config.json"

def get_embeddings():
    provider = os.getenv("EMBEDDINGS_PROVIDER", "openai").lower()
    model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
    if provider == "hf":
        return "hf", model, HuggingFaceEmbeddings(model_name=model)
    return "openai", model, OpenAIEmbeddings(model=model)

def load_documents():
    loaders = [
        DirectoryLoader(str(DOCS_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(str(DOCS_DIR), glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(str(DOCS_DIR), glob="**/*.md", loader_cls=TextLoader),
        DirectoryLoader(str(DOCS_DIR), glob="**/*.docx", loader_cls=Docx2txtLoader),
    ]
    docs = []
    for l in loaders:
        docs.extend(l.load())
    return docs

def main():
    load_dotenv()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    provider, model, embedder = get_embeddings()
    vs = FAISS.from_documents(chunks, embedder)
    vs.save_local(str(DATA_DIR / "faiss_index"))
    with open(CONFIG_PATH, "w") as f:
        json.dump({"provider": provider, "model": model}, f)

if __name__ == "__main__":
    main()
