import os, json
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "faiss_index"
CONFIG_PATH = DATA_DIR / "embedding_config.json"

def get_embeddings_from_config():
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    if cfg["provider"] == "hf":
        return HuggingFaceEmbeddings(model_name=cfg["model"])
    return OpenAIEmbeddings(model=cfg["model"])

def main():
    load_dotenv()
    st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")
    st.title("📚 RAG Document Search")
    embedder = get_embeddings_from_config()
    vs = FAISS.load_local(str(INDEX_DIR), embedder, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    if "history" not in st.session_state:
        st.session_state.history = []

    user_q = st.chat_input("Ask about your documents...")
    if user_q:
        docs = retriever.get_relevant_documents(user_q)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = ChatPromptTemplate.from_template(
            "Use this context to answer:\n{context}\n\nQuestion: {question}"
        )
        resp = llm.invoke(prompt.format_messages(context=context, question=user_q))
        st.session_state.history.append(("user", user_q))
        st.session_state.history.append(("assistant", resp.content))

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

if __name__ == "__main__":
    main()
