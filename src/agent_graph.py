## ** LangGraph Retrieval Agent Workflow ** ##
import os
from typing import TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    question: str
    documents: List[Document]
    generation: str

def create_card_analyst_graph(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model = "gemini-1.5-flash",
        temperature = 0.2,
        google_api_key=api_key
    )
    def retrieve_node(state: AgentState):
        question = state["question"]
        docs = retriever.invoke(question)
        return {"documents" : docs}
    
    def generate_node(state: AgentState):
        question = state["question"]
        docs = state["documents"]

        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""You are CardAnalyst_AI, an expert financial and credit card rewards analyst.
Use the provided context to answer the user's question precisely. Highlight rewards structures, fees, or compliance rules where applicable.

Context:
{context}

Question: {question}
Answer:"""
        response = llm.invoke(prompt)
        return {"generation": response.content}

    # Build LangGraph workflow
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

    

