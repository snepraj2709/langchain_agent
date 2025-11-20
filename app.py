import streamlit as st
from agent import create_agent
import os

st.title("🤖 Agentic Pipeline - Weather or RAG")

# Initialize agent
if "agent" not in st.session_state:
    pdf_path = "sample.pdf"
    if not os.path.exists(pdf_path):
        st.error(f"Please add a PDF file named '{pdf_path}' to the project directory")
        st.stop()
    
    st.session_state.agent = create_agent(pdf_path)
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if query := st.chat_input("Ask about weather or PDF content..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.agent.invoke({
                "query": query,
                "route": "",
                "context": "",
                "response": ""
            })
            
            response = result["response"]
            st.write(response)
            
            # Show route for debugging
            st.caption(f"Route: {result['route']}")
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})