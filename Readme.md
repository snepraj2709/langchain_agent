# LangGraph Agentic Pipeline - Weather & PDF RAG

A simple agentic system that routes queries to either fetch weather data or answer questions from a PDF using RAG.

## Setup

### 1. Install Dependencies

```bash
pip install langchain langchain-openai langchain-community langgraph qdrant-client pypdf requests streamlit python-dotenv pytest langchain_text_splitters
```

### 2. Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=weather-rag-agent
```

## Project Structure

```
.
├── .env
├── agent.py          # LangGraph agent logic
├── weather.py        # Weather API handler
├── rag.py           # RAG implementation
├── app.py           # Streamlit UI
├── test_agent.py    # Test cases
├── sample.pdf       # Sample PDF document
└── README.md        # This file
```

## Running the Application

### 1. Create Sample PDF

Create a `sample.pdf` in the root with some content for testing.

### 2. Run Tests

```bash
pytest test_agent.py -v
```

### 3. Start Streamlit App

```bash
streamlit run app.py
```

### 4. Test Queries

**Weather queries:**
- "What's the weather in London?"
- "Tell me the temperature in Paris"

**PDF queries:**
- "What is mentioned in the document?"
- "Summarize the content"

## How It Works

1. **Router Node**: LLM determines if query is about weather or PDF content
2. **Weather Node**: Extracts city name and calls OpenWeatherMap API
3. **PDF Node**: Retrieves relevant chunks from vector database using RAG
4. **Response Node**: LLM generates final answer based on context
5. **LangSmith**: Automatically traces all LLM calls for evaluation

## Architecture Flow

```
User Query → Router → [Weather API | PDF RAG] → Response Generator → User
```

## Key Components

- **LangGraph**: Orchestrates the agentic workflow
- **OpenWeatherMap**: Provides real-time weather data
- **Qdrant**: Vector database for embeddings (in-memory)
- **LangChain**: LLM integration and document processing
- **LangSmith**: Automatic tracing and evaluation
- **Streamlit**: Simple chat interface

## Notes

- All LLM calls are traced in LangSmith automatically via environment variables
- Vector database runs in-memory (no external setup needed)
- Code is minimal and modular
- Each component is independently testable