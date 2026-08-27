from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel



from dotenv import load_dotenv
load_dotenv()


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)  
retriever = vectorstore.as_retriever(search_kwargs={"k":4})


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


parser = StrOutputParser()


prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know.

context:
{context}

Question: {question}
""")


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


parallel_chain = RunnableParallel({
    "context": retriever | format_docs,
    "question": RunnablePassthrough()
})


main_chain = parallel_chain | prompt | llm | parser


if __name__ == "__main__":
    question = "What did the president say about healthcare?"
    answer = main_chain.invoke(question)
    print(answer)

    docs = retriever.invoke(question)
    print("\n--- Sources used ---")
    for d in docs:
        print(d.page_content[:150], "...\n")
