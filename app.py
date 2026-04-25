# Initializes simple front-end

import streamlit as st
from query import ask

st.title("RAG CS Tutor")

question = st.text_input("Ask a CS Question:")
if question:
    answer, sources = ask(question)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for s in sources:
        st.write("-", s)