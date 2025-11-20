from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from weather import get_weather
from rag import RAGSystem
import os
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    query: str
    route: str
    context: str
    response: str

def router_node(state: AgentState) -> AgentState:
    """Decide whether to route to weather API or PDF RAG"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a routing assistant. Analyze the user's query and determine the appropriate route.

Rules:
- If the query asks about weather, temperature, climate, or mentions a city's weather conditions, respond with: weather
- If the query asks about document content, PDF information, or what's mentioned in a document, respond with: pdf
- Respond with ONLY one word: either 'weather' or 'pdf'

Examples:
- "What's the weather in London?" -> weather
- "Tell me about the temperature in Paris" -> weather
- "What is mentioned in the document?" -> pdf
- "Summarize the PDF content" -> pdf"""),
        ("user", "{query}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"query": state["query"]})
    
    route = result.content.strip().lower()
    
    # Check for explicit matches
    if route == "weather" or "weather" in route:
        state["route"] = "weather"
    elif route == "pdf" or "pdf" in route or "document" in route:
        state["route"] = "pdf"
    else:
        # Default to pdf for document-related queries
        state["route"] = "pdf"
    
    print(f"[DEBUG] Query: {state['query']}")
    print(f"[DEBUG] LLM Response: {route}")
    print(f"[DEBUG] Chosen Route: {state['route']}")
    
    return state

def weather_node(state: AgentState) -> AgentState:
    """Handle weather queries"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Extract city from query
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the city name from the query. Reply with only the city name."),
        ("user", "{query}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"query": state["query"]})
    city = result.content.strip()
    
    # Fetch weather
    weather_data = get_weather(city)
    state["context"] = str(weather_data)
    
    return state

def pdf_node(state: AgentState, rag_system: RAGSystem) -> AgentState:
    """Handle PDF queries using RAG"""
    # Retrieve relevant chunks
    relevant_docs = rag_system.retrieve(state["query"])
    state["context"] = "\n\n".join(relevant_docs)
    
    print(f"[DEBUG] Retrieved {len(relevant_docs)} chunks")
    
    return state

def response_node(state: AgentState) -> AgentState:
    """Generate final response using LLM"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's query based on the provided context. Be concise and helpful."),
        ("user", "Query: {query}\n\nContext: {context}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"query": state["query"], "context": state["context"]})
    
    state["response"] = result.content
    
    return state

def route_decision(state: AgentState) -> Literal["weather", "pdf"]:
    """Conditional edge routing"""
    return state["route"]

def create_agent(pdf_path: str):
    """Create the LangGraph agent"""
    rag_system = RAGSystem(pdf_path)
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("weather", weather_node)
    workflow.add_node("pdf", lambda state: pdf_node(state, rag_system))
    workflow.add_node("response", response_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "weather": "weather",
            "pdf": "pdf"
        }
    )
    
    # Add edges to response
    workflow.add_edge("weather", "response")
    workflow.add_edge("pdf", "response")
    workflow.add_edge("response", END)
    
    return workflow.compile()