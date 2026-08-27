#-------------------  Ingezst + Chunk  -----------------------

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = TextLoader("state_of_the_union.txt", encoding="utf-8")
documents = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

chunks = splitter.split_documents(documents)


print(f"Total chunks: {len(chunks)}")
print(chunks[5].page_content)