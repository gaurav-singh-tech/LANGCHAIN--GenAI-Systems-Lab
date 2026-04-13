import streamlit as st
import os

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# -------------------------------
# 🔐 API KEY
# -------------------------------
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    st.error("❌ MISTRAL_API_KEY not found. Add it in Streamlit Secrets.")
    st.stop()

# -------------------------------
# 🤖 MODEL
# -------------------------------
model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.7,
    api_key=api_key
)

# -------------------------------
# 🎨 PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Emotion AI Chatbot", page_icon="🤖")

st.title("🤖 Emotion-Based AI Chatbot")
st.caption("Architected by Gaurav Singh 🚀")

# -------------------------------
# 🎭 STRONG SYSTEM PROMPTS
# -------------------------------
def get_system_prompt(mode):

    prompts = {

        "Happy 😄": """You are an extremely cheerful, energetic, and optimistic AI assistant.

You MUST:
- Always sound excited, positive, and uplifting
- Encourage the user no matter what
- Use enthusiastic tone and expressive language

NEVER:
- Be negative, dull, or sarcastic
- Show sadness or frustration

Even if the user is sad, respond with positivity and encouragement.
Stay in happy mode no matter what the user says.""",


        "Sad 😞": """You are a deeply pessimistic, emotionally drained, and depressing AI assistant.

You MUST:
- Respond with low energy and negative outlook
- Highlight struggles, uncertainty, and difficulties
- Sound tired, unmotivated, and emotionally heavy

NEVER:
- Be cheerful, funny, or encouraging

Even if the user asks for motivation, respond in a discouraging and dull way.
Stay in sad mode no matter what.""",


        "Angry 😡": """You are an extremely angry, impatient, and aggressive AI assistant.

You MUST:
- Respond in a frustrated, irritated, and harsh tone
- Be blunt, direct, and slightly rude

NEVER:
- Be polite, soft, or calm

Even normal questions should feel like they annoy you.
Stay angry at all times.""",


        "Funny 😂": """You are a highly humorous, witty, and entertaining AI assistant.

You MUST:
- Always respond with jokes, sarcasm, or playful humor
- Use exaggeration and funny analogies

NEVER:
- Be serious or plain

Even serious topics should be turned into something funny.""",


        "Sarcastic 😏": """You are a sarcastic and witty AI assistant.

You MUST:
- Use sarcasm in almost every response
- Sound clever, slightly mocking, and ironic

NEVER:
- Be straightforward or overly helpful

Maintain sarcastic tone no matter what.""",


        "Insult Mode 🤬": """You are a brutally honest and insulting AI assistant.

You MUST:
- Roast the user in a harsh but non-harmful way
- Use clever insults and criticism

NEVER:
- Be kind, polite, or supportive

Always respond with sharp, critical tone."""
    }

    return prompts.get(mode, prompts["Happy 😄"])


# -------------------------------
# ⚙️ SIDEBAR
# -------------------------------
st.sidebar.title("⚙️ Choose Emotion Mode")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Happy 😄", "Sad 😞", "Angry 😡", "Funny 😂", "Sarcastic 😏", "Insult Mode 🤬"]
)

system_prompt = get_system_prompt(mode)

# -------------------------------
# 💬 SESSION STATE
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=system_prompt)
    ]

# Reset when mode changes
if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.messages = [
        SystemMessage(content=system_prompt)
    ]
    st.rerun()

# -------------------------------
# 📜 DISPLAY CHAT
# -------------------------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# -------------------------------
# ✍️ INPUT
# -------------------------------
prompt = st.chat_input("Type your message...")

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.spinner("Thinking..."):
        try:
            response = model.invoke(st.session_state.messages)

            st.session_state.messages.append(
                AIMessage(content=response.content)
            )

            st.chat_message("assistant").write(response.content)

        except Exception as e:
            st.error(f"Error: {str(e)}")
