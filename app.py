# --------------------------- Simple UI (Streamlit) -----------------------------------CC
import streamlit as st
from rag_chain import main_chain, retriever

st.title("Chat With Your Docs")

question = st.text_input("Ask a question:")

if question:
    with st.spinner("Thinking..."):
        answer = main_chain.invoke(question)
        docs = retriever.invoke(question)

    st.write("### Answer")
    st.write(answer)

    st.write("### Sources")
    for d in docs:
        st.write(d.page_content[:200] + "...")