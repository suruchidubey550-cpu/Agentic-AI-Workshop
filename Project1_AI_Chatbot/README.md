# 🤖 My AI Chatbot (Project 1 of 4)
Built by **Your Name** at the VGU × Kvon Tech GenAI & Agentic AI Workshop, 28 October 2026.

A chatbot app with a personality you choose (or write yourself), chat memory, a creativity slider and live typing.

## How it works
- **App:** Streamlit (`app.py`), with the personalities in `personas.py`
- **AI model:** qwen3:4b-instruct-2507-q4_K_M, running on Ollama, called through the OpenAI-style chat API
- **What I learned:** prompts, system prompts, temperature, chat memory, streaming, building an AI app

## How to run it
1. In Google Colab (with a T4 GPU), run the setup cell (0.1) from the Day 3 notebook.
2. Upload `app.py` and `personas.py` to Colab's Files panel.
3. Run `launch_app("app.py")` and open the link.
