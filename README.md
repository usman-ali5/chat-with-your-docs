# Chat With Your Docs — RAG Q&A System

A Retrieval-Augmented Generation (RAG) pipeline that answers questions grounded in a source document, using LangChain, OpenAI, and ChromaDB. Built with a FastAPI backend and a Streamlit UI.

## Architecture

```
Documents → Chunking → Embeddings → Chroma (vector DB)
                                          ↓
Question → Retriever → Prompt + LLM → Answer + Sources
                                          ↓
                                  FastAPI / Streamlit
```

## Tech Stack
- LangChain (chains, retrievers, prompt templates)
- OpenAI (`text-embedding-3-small`, `gpt-4o-mini`)
- ChromaDB (vector storage)
- FastAPI (API layer)
- Streamlit (UI layer)

## How It Works
1. `ingest.py` — loads and splits the source document into chunks
2. `embed_store.py` — embeds chunks and stores them in Chroma
3. `rag_chain.py` — retrieves relevant chunks and generates a grounded answer using an LLM
4. `main.py` — exposes the pipeline as a REST API
5. `app.py` — simple Streamlit chat interface
6. `eval.py` — basic retrieval sanity checks

## Setup
\`\`\`bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
\`\`\`

Create a `.env` file with:
\`\`\`
OPENAI_API_KEY=your_key_here
\`\`\`

## Run
\`\`\`bash
python ingest.py
python embed_store.py
python rag_chain.py        # test in terminal
uvicorn main:app --reload  # API at localhost:8000/docs
streamlit run app.py       # UI at localhost:8501
\`\`\`

## Example
**Q:** What did the speech say about jobs?
**A:** The speech highlighted 6.5 million new jobs created in the past year, with 369,000 in manufacturing...

## Known Limitations
- Currently single-document (multi-document support planned)
- Basic keyword-based retrieval evaluation, not full RAGAS scoring yet