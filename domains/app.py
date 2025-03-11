import streamlit as st
from domains.injestion.doc_loader import file_loader
from domains.agents.routes import react_orchestrator
from domains.retreival.routes import run_rag


st.title("RAG with Agents")
st.