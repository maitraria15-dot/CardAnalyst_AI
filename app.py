import os
import streamlit as st
from dotenv import load_dotenv
from src.parser import load_and_split_pdf
from src.vector_store import create_vector_store, save_vector_store, load_vector_store
from src.agent_graph import create_card_analyst_graph

load_dotenv()

# Set page config for a clean wide layout
st.set_page_config(page_title="CardAnalyst AI", page_icon="💳", layout="wide")

# Custom CSS injection for sleek styling, cards, and clean typography
st.markdown("""
    <style>
    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header Container */
    .header-container {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    
    /* Title typography */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .brand-ai {
        color: #2563eb;
    }
    
    .author-badge {
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748b;
        background: #e2e8f0;
        padding: 4px 10px;
        border-radius: 20px;
        margin-left: 12px;
        vertical-align: middle;
        display: inline-block;
    }
    
    .subtitle {
        font-size: 1.0rem;
        color: #475569;
        margin-top: 8px;
        margin-bottom: 0;
        font-weight: 400;
    }
    </style>
""", unsafe_allow_html=True)

# Render Custom Executive Header
st.markdown("""
    <div class="header-container">
        <h1 class="main-title">
            CardAnalyst <span class="brand-ai">AI</span>
            <span class="author-badge">crafted by Ria Maitra</span>
        </h1>
        <p class="subtitle">
            Autonomous Credit Card Document Intelligence & Competitive Analysis Suite
        </p>
    </div>
""", unsafe_allow_html=True)

# Side bar for configuration & document upload
st.sidebar.header("📁 Document Ingestion")
uploaded_files = st.sidebar.file_uploader(
    "Upload Credit Card Agreements (PDFs)", 
    type=["pdf"], 
    accept_multiple_files=True
)

db_path = "vector_db"

if uploaded_files:
    os.makedirs("data", exist_ok=True)
    
    if st.sidebar.button("Process All Documents"):
        with st.spinner("Parsing documents and building vector embeddings..."):
            try:
                all_chunks = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join("data", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    chunks = load_and_split_pdf(file_path)
                    all_chunks.extend(chunks)
                
                vstore = create_vector_store(all_chunks)
                save_vector_store(vstore, db_path)
                st.sidebar.success(f"Successfully indexed {len(uploaded_files)} documents!")
                
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    st.sidebar.error("⚠️ Free tier rate limit reached. Try uploading fewer files or wait 30 seconds.")
                else:
                    st.sidebar.error(f"An error occurred: {e}")

# Main UI query interface inside a neat card-like layout
st.markdown("### 💬 Ask Financial Assistant")
query = st.text_input("Query policy, rewards, or fee schedules (e.g., 'Compare annual fees and purchase APRs'):", label_visibility="collapsed", placeholder="Ask a question about the credit card policy, rewards, or fees...")

if query:
    if not os.path.exists(db_path):
        st.warning("Please upload and process credit card agreement PDFs first using the sidebar.")
    else:
        with st.spinner("Analyzing portfolio document vector space..."):
            vstore = load_vector_store(db_path)
            app_graph = create_card_analyst_graph(vstore)

            result = app_graph.invoke({"question": query})

            st.markdown("---")
            st.markdown("## 📊 Analysis Result")
            
            # Safely extract text from graph output and render as formatted Markdown
            generation_output = result.get("generation", "")
            
            if isinstance(generation_output, list) and len(generation_output) > 0:
                answer_text = generation_output[0].get("text", str(generation_output))
            elif isinstance(generation_output, dict):
                answer_text = generation_output.get("text", str(generation_output))
            else:
                answer_text = str(generation_output)
                
            st.markdown(answer_text)

            with st.expander("🔍 View Retrieved Source Chunks"):
                for i, doc in enumerate(result.get("documents", [])):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)