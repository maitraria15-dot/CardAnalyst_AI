## ** Streamlit Interface ** ##

import os
import streamlit as st
from dotenv import load_dotenv
from src.parser import load_and_split_pdf
from src.vector_store import create_vector_store, save_vector_store, load_vector_store
from src.agent_graph import create_card_analyst_graph

load_dotenv()

st.set_page_config(page_title="CardAnalyst_AI", layout = "wide")

st.title("CardAnalyst_AI")
st.markdown("### Autonomous Credit Card Document Intelligence & Competitive Analysis")

# Side bar for configuration & document upload
st.sidebar.header("Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Credit card Agreement (PDF)", type = ["pdf"])

db_path = "vector_db"

if uploaded_file is not None:
    os.makedirs("data", exist_ok = True)
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.writ(uploaded_file.getbuffer())

    if st.sidebar.button("Process Document"):
        with st.spinner("Parsing document and building vector embeddings..."):
            chunks = load_and_split_pdf(file_path)
            vstore = create_vector_store(chunks)
            save_vector_store(vstore, db_path)
            st.sidebar.success("Document processed and indexed successfully!")

# Main UI chat / query interface
query = st.text_input("Ask a question about the credit card policy, rewards, or fees:")

if query:
    if not os.path.exists(db_path):
        st.warning("Please upload and process a credit card agreement PDF first using the sidebar.")
    else:
        with st.spinner("Analyzing portfolio document..."):
            vstore = load_vector_store(db_path)
            app_graph = create_card_analyst_graph(vstore)

            result = app_graph.invoke({"question": query})

            st.markdown("## Analysis Result")
            st.write(result["generation"])

            with st.expander("View Retrieved Source Chunks"):
                for i, doc in enumerate(result["documents"]):
                    st.markdown(f"**Chunks {i+1}:**")
                    st.write(doc.page_content)
                    