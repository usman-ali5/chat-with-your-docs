# --------------------  Embed + Store in Chroma ----------------

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from ingest import chunks    # resue chunks from ingest.py

from dotenv import load_dotenv
load_dotenv()


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)


print("Stored", vectorstore._collection.count(), "chunks in chroma")


results = vectorstore.similarity_search("What did the president say about the jobs?", k=3)
for r in results:
    print("---")
    print(r.page_content)