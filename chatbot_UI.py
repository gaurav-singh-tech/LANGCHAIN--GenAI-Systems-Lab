import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Initialize model
model = ChatMistralAI(model="mistral-small-2506", temperature=0.6)

st.title("😂 Funny AI Chatbot")
st.write("Chat with a very sad and irritating AI agent!")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a very sad and irritating AI agent")
    ]

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# User input box
prompt = st.chat_input("Type your message...")

if prompt:
    
    # Add user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    # Get response from model
    response = model.invoke(st.session_state.messages)

    # Save AI response
    st.session_state.messages.append(AIMessage(content=response.content))

    # Display response
    st.chat_message("assistant").write(response.content)