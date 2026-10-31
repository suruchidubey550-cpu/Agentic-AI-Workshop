# 6.1 VGU Buddy: a chatbot that answers only from VGU's website, with sources
import glob, os, re
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer

client = OpenAI(base_url=os.environ["VGU_BASE_URL"], api_key=os.environ["VGU_API_KEY"])
MODEL = os.environ["VGU_MODEL"]
NO_ANSWER = "I don't know: that isn't in the VGU information I have."
RULES = ("You are VGU Buddy, a helpful assistant for students of Vivekananda Global University (VGU), Jaipur. "
         "Answer ONLY from the numbered pieces of VGU's website below. "
         'If they do not contain the answer, reply exactly: "' + NO_ANSWER + '" '
         "After each fact, write the number of the piece it came from in square brackets, like [2]. "
         "Keep it under 80 words.")


@st.cache_resource                                   # runs once when the app starts, then is reused
def load_knowledge():
    with open(sorted(glob.glob("vgu_knowledge*.md"))[0], encoding="utf-8") as f:
        document = f.read()
    pieces, section = [], ""
    for paragraph in document.split("\n\n"):         # the same cutting as cell 2.1
        paragraph = paragraph.strip()
        if paragraph.startswith("## "):
            section = paragraph[3:]
        elif "(Source:" in paragraph:
            text, _, source = paragraph.rpartition("(Source:")
            pieces.append({"text": section + ": " + text.strip(), "sources": re.findall(r"https://[^\s;)]+", source)})
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return pieces, embedder, embedder.encode([p["text"] for p in pieces], normalize_embeddings=True)


def search(question, k):                             # the same meaning search as cell 3.4
    pieces, embedder, vectors = load_knowledge()
    question = re.sub(r"\b(vgu|vivekananda global university)('s)?\b", " ", question, flags=re.I)
    scores = vectors @ embedder.encode(question, normalize_embeddings=True)
    return [(float(scores[i]), pieces[i]) for i in scores.argsort()[::-1][:k]]


def sources_of(answer, found):                       # piece numbers like [2] in the answer → source links
    links = []
    for n in re.findall(r"\[(\d+)\]", answer):
        if 1 <= int(n) <= len(found):
            links += [url for url in found[int(n) - 1][1]["sources"] if url not in links]
    return links


st.set_page_config(page_title="VGU Buddy", page_icon="🎓")
st.title("🎓 VGU Buddy")
st.caption("Answers only from VGU's official website (vgu.ac.in, collected 25 Oct 2026) · Project 2 · built by me!")

with st.sidebar:
    k = st.slider("Pieces to read before answering (k)", 2, 8, 4)
    show_pieces = st.toggle("Show what it read", value=True)
    if st.button("🧹 New chat"):
        st.session_state.messages = []
    st.caption(f"Model: {MODEL}")

if "messages" not in st.session_state:
    st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask about VGU: fees, hostels, clubs, placements..."):
    question = question.strip()[:500]
    if not question:
        st.stop()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    found = search(question, k)
    numbered = "\n\n".join(f"[{n}] {piece['text']}" for n, (score, piece) in enumerate(found, start=1))
    messages = [{"role": "system", "content": RULES},
                {"role": "user", "content": f"Pieces of VGU's website:\n{numbered}\n\nQuestion: {question}"}]
    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(model=MODEL, messages=messages, temperature=0, stream=True)
            answer = st.write_stream(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
        except Exception as error:                  # e.g. the AI server stopped
            st.error(f"The AI didn't answer ({type(error).__name__}). Check the setup cell, then try again.")
            st.stop()
        links = sources_of(answer, found)
        if links:
            sources_line = "📚 **Sources:** " + " · ".join(f"[{url.split('vgu.ac.in')[-1] or '/'}]({url})" for url in links)
            st.markdown(sources_line)
            answer += "\n\n" + sources_line
        if show_pieces:
            with st.expander("🔎 What it read"):
                for n, (score, piece) in enumerate(found, start=1):
                    st.markdown(f"**[{n}]** similarity {score:.2f} · {piece['text'][:300]}...")
    st.session_state.messages.append({"role": "assistant", "content": answer})
