import pytest
from weather import get_weather
from rag import RAGSystem
from agent import create_agent
import os

def test_weather_api():
    """Test weather API handling"""
    result = get_weather("London")
    
    assert "city" in result or "error" in result
    if "city" in result:
        assert result["city"] == "London"
        assert "temperature" in result
        assert "description" in result

def test_rag_retrieval():
    """Test RAG retrieval logic"""
    # Create a temporary test PDF
    pdf_path = "sample.pdf"
    
    if not os.path.exists(pdf_path):
        pytest.skip("sample.pdf not found")
    
    rag = RAGSystem(pdf_path)
    results = rag.retrieve("test query")
    
    assert isinstance(results, list)
    assert len(results) > 0
    
    # Test collection info
    info = rag.get_collection_info()
    assert "vectors_count" in info
    assert info["vectors_count"] > 0

def test_agent_weather_route():
    """Test agent routing to weather"""
    pdf_path = "sample.pdf"
    
    if not os.path.exists(pdf_path):
        pytest.skip("sample.pdf not found")
    
    agent = create_agent(pdf_path)
    result = agent.invoke({
        "query": "What's the weather in Paris?",
        "route": "",
        "context": "",
        "response": ""
    })
    
    assert result["route"] == "weather"
    assert len(result["response"]) > 0

def test_agent_pdf_route():
    """Test agent routing to PDF"""
    pdf_path = "sample.pdf"
    
    if not os.path.exists(pdf_path):
        pytest.skip("sample.pdf not found")
    
    agent = create_agent(pdf_path)
    result = agent.invoke({
        "query": "What is mentioned in the document?",
        "route": "",
        "context": "",
        "response": ""
    })
    
    assert result["route"] == "pdf"
    assert len(result["response"]) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])