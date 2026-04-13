import streamlit as st
import os

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# -------------------------------
# 🔐 API KEY FROM STREAMLIT SECRETS
# -------------------------------
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    st.error("❌ MISTRAL_API_KEY not found. Add it in Streamlit Secrets.")
    st.stop()

# -------------------------------
# 🤖 MODEL INITIALIZATION
# -------------------------------
try:
    model = ChatMistralAI(
        model="mistral-small-2506",
        temperature=0.6,
        api_key=api_key
    )
except Exception as e:
    st.error(f"Model Initialization Error: {e}")
    st.stop()

# -------------------------------
# 🎨 PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

st.title("🤖 AI Chatbot")
st.write("Chat with different AI personalities")

# -------------------------------
# ⚙️ SIDEBAR SETTINGS
# -------------------------------
st.sidebar.title("⚙️ Settings")

mode = st.sidebar.selectbox(
    "Choose AI Personality",
    ["Funny 😄", "Sad 😒"]
)

if mode == "Funny 😄":
    system_prompt = "You are a funny AI agent"
else:
    system_prompt = "You are a very sad and irritating AI agent"

# -------------------------------
# 💬 SESSION STATE (CHAT MEMORY)
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=system_prompt)
    ]

# Reset chat button
if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.messages = [
        SystemMessage(content=system_prompt)
    ]
    st.rerun()

# -------------------------------
# 📜 DISPLAY CHAT HISTORY
# -------------------------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# -------------------------------
# ✍️ USER INPUT
# -------------------------------
prompt = st.chat_input("Type your message...")

if prompt:
    # Save user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    # Generate response
    with st.spinner("Thinking... 🤔"):
        try:
            response = model.invoke(st.session_state.messages)

            # Save AI response
            st.session_state.messages.append(
                AIMessage(content=response.content)
            )

            # Show response
            st.chat_message("assistant").write(response.content)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
