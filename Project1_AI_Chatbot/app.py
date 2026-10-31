# 6.1 The AI Chatbot app: a real Python file that Streamlit runs as a website
import os
import streamlit as st
from openai import OpenAI
from personas import PERSONAS

# Connect to the AI through its API (the setup cell put these settings in environment variables)
client = OpenAI(base_url=os.environ["VGU_BASE_URL"], api_key=os.environ["VGU_API_KEY"])
MODEL = os.environ["VGU_MODEL"]

st.set_page_config(page_title="My AI Chatbot", page_icon="🤖")
st.title("🤖 My AI Chatbot")
st.caption("VGU × Kvon Tech GenAI Workshop · Project 1 · built by me!")

# The sidebar holds the settings
with st.sidebar:
    persona_name = st.selectbox("Personality", list(PERSONAS))
    custom = st.text_area("…or write your own personality", placeholder="You are a ...")
    temperature = st.slider("Creativity (temperature)", 0.0, 1.5, 0.7, 0.1)
    if st.button("🧹 New chat"):
        st.session_state.messages = []
    st.caption(f"Model: {MODEL}")

# Memory: st.session_state survives every click, so the chat history lives there
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Say something..."):
    prompt = prompt.strip()[:4000]                  # very long messages are cut, not crashed
    if not prompt:
        with st.chat_message("assistant"):
            st.markdown("Type something first! 🙂")
        st.stop()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    persona = custom.strip() or PERSONAS[persona_name]
    messages = [{"role": "system", "content": persona}] + st.session_state.messages
    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(model=MODEL, messages=messages,
                                                    temperature=temperature, stream=True)
            reply = st.write_stream(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
        except Exception as error:                  # e.g. the server stopped, or a cloud API's rate limit
            st.error(f"The AI didn't answer ({type(error).__name__}). Check the setup cell, then try again.")
            st.stop()
    st.session_state.messages.append({"role": "assistant", "content": reply})
