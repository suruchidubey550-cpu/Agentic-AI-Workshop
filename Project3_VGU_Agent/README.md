# 🛠️ VGU Agent (Project 3 of 4)
Built by **Your Name** at the VGU × Kvon Tech GenAI & Agentic AI Workshop, 30 October 2026.

An **AI agent** for VGU students. It decides by itself which tools to use, shows every step, and asks before
saving anything.

## Tools
- 📅 today's date · 🧮 an exact, safe calculator · 🌦️ live weather (Open-Meteo) · 📖 Wikipedia
- 🎓 VGU Buddy (Project 2): answers from VGU's official website
- 📝 reminders saved in a SQLite database: **the student must approve** every save

## How it works
The **agent loop**: qwen3:4b-instruct-2507-q4_K_M (on Ollama) reads the question and a "menu" of tools (function calling), asks for a
tool, the app runs it and sends back the result, and this repeats (up to a step limit) until it can answer.

## How to run it
1. In Google Colab (T4 GPU), run the setup cell (0.1) from the Day 5 notebook.
2. Upload `app.py` and `vgu_knowledge.md` to Colab's Files panel.
3. Run `launch_app("app.py")` and open the link.
