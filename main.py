# ----------------------------- Wrap in FastAPI -----------------------------
from fastapi import FastAPI
from pydantic import BaseModel
from  rag_chain import main_chain, retriever


app = FastAPI()


class Question(BaseModel):
    query: str


@app.post("/ask")
def ask(q: Question):
    answer = main_chain.invoke(q.query)
    docs = retriever.invoke(q.query)
    sources = [d.page_content[:150] for d in docs]
    return {"answer": answer, "sources": sources}


@app.get("/health")
def health():
    return{"status": "ok"}
