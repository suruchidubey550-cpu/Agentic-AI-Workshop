# 🎓 VGU Buddy (Project 2 of 4)
Built by **Your Name** at the VGU × Kvon Tech GenAI & Agentic AI Workshop, 29 October 2026.

A chatbot that answers questions about Vivekananda Global University **only from VGU's official website**
(collected on 25 October 2026), shows its **sources**, and says "I don't know" when the answer isn't there.

## How it works (RAG: Retrieval-Augmented Generation)
1. **Cut** the knowledge file into 60 pieces, each with its source page
2. **Embed** every piece with `all-MiniLM-L6-v2` (meaning → 384 numbers)
3. **Search:** find the pieces closest in meaning to the question
4. **Answer:** qwen3:4b-instruct-2507-q4_K_M (on Ollama) answers only from those pieces and cites them like [2]
- Tested with an 18-question exam (15 answerable + 3 "I don't know" questions)

## How to run it
1. In Google Colab (T4 GPU), run the setup cell (0.1) from the Day 4 notebook.
2. Upload `app.py` and `vgu_knowledge.md` to Colab's Files panel.
3. Run `launch_app("app.py")` and open the link.
