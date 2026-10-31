# 6.1 VGU Agent: an AI that chooses tools, shows its steps, and asks before saving anything
import datetime, glob, json, os, re, sqlite3
import requests
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer

client = OpenAI(base_url=os.environ["VGU_BASE_URL"], api_key=os.environ["VGU_API_KEY"])
MODEL = os.environ["VGU_MODEL"]
HEADERS = {"User-Agent": "VGU-GenAI-Workshop/1.0 (student project)"}


# ---------- VGU Buddy (Day 4), loaded once
@st.cache_resource
def load_vgu():
    with open(sorted(glob.glob("vgu_knowledge*.md"))[0], encoding="utf-8") as f:
        document = f.read()
    pieces, section = [], ""
    for paragraph in document.split("\n\n"):
        paragraph = paragraph.strip()
        if paragraph.startswith("## "):
            section = paragraph[3:]
        elif "(Source:" in paragraph:
            pieces.append(section + ": " + paragraph.rpartition("(Source:")[0].strip())
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return pieces, embedder, embedder.encode(pieces, normalize_embeddings=True)


def ask_vgu_buddy(question, k=8):
    pieces, embedder, vectors = load_vgu()
    search_text = re.sub(r"\b(vgu|vivekananda global university)('s)?\b", " ", question, flags=re.I)
    scores = vectors @ embedder.encode(search_text, normalize_embeddings=True)
    numbered = "\n\n".join(f"[{n}] {pieces[i]}" for n, i in enumerate(scores.argsort()[::-1][:k], start=1))
    rules = ("Answer ONLY from the numbered pieces of VGU's website below. If they do not contain the answer, "
             "say you don't know. Keep it under 80 words.")
    return client.chat.completions.create(model=MODEL, temperature=0, messages=[
        {"role": "system", "content": rules},
        {"role": "user", "content": f"Pieces of VGU's website:\n{numbered}\n\nQuestion: {question}"}]).choices[0].message.content


# ---------- the other tools (the same as the notebook's Part 2)
def get_today():
    return datetime.date.today().strftime("Today is %A, %d %B %Y.")


def calculator(expression):
    text = str(expression).replace(",", "").replace("×", "*")
    text = re.sub(r"([\d.]+)\s*%\s*of\s*", r"(\1/100)*", text)
    text = re.sub(r"([\d.]+)\s*%", r"(\1/100)", text)
    if not re.fullmatch(r"[\d+\-*/(). ]+", text) or "**" in text or len(text) > 200:
        return "Error: the calculator only understands numbers and + - * / ( )"
    return f"{eval(text, {'__builtins__': {}}):,.2f}".rstrip("0").rstrip(".")   # safe: only digits and + - * / ( ) left


def get_weather(city):
    found = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1},
                         headers=HEADERS, timeout=15).json()
    if not found.get("results"):
        return f"Error: no city called {city!r} was found."
    place = found["results"][0]
    now = requests.get("https://api.open-meteo.com/v1/forecast", headers=HEADERS, timeout=15, params={
        "latitude": place["latitude"], "longitude": place["longitude"],
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"}).json()["current"]
    return (f"{place['name']}: {now['temperature_2m']} °C, humidity {now['relative_humidity_2m']}%, "
            f"rain {now['precipitation']} mm, wind {now['wind_speed_10m']} km/h")


def search_wikipedia(topic):
    hits = requests.get("https://en.wikipedia.org/w/api.php", headers=HEADERS, timeout=15, params={
        "action": "query", "list": "search", "srsearch": topic, "srlimit": 1, "format": "json"}).json()["query"]["search"]
    if not hits:
        return f"No Wikipedia article found for {topic!r}."
    page = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/" + hits[0]["title"].replace(" ", "_"),
                        headers=HEADERS, timeout=15).json()
    return f"{hits[0]['title']}: {page.get('extract', '')[:600]}"


# ---------- reminders: long-term memory in a database, and the agent must ASK before saving
def database():
    db = sqlite3.connect("reminders.db")
    db.execute("CREATE TABLE IF NOT EXISTS reminders (text TEXT, saved TEXT)")
    return db


def add_reminder(text):                              # doesn't save: it puts the reminder on the "ask the student" list
    st.session_state.pending.append(text)
    return "Not saved yet: the student must click Save in the app first. Tell them to approve it below."


def list_reminders():
    with database() as db:
        rows = db.execute("SELECT text, saved FROM reminders").fetchall()
    return "; ".join(f"{text} (saved {saved})" for text, saved in rows) or "No reminders saved yet."


def describe(name, description, **inputs):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": {key: {"type": "string", "description": text} for key, text in inputs.items()},
        "required": list(inputs)}}}


TOOLBOX = {  # name: (function, menu entry, label in the sidebar)
    "get_today": (get_today, describe("get_today", "Get today's date and day of the week."), "📅 Today's date"),
    "calculator": (calculator, describe("calculator", "Do exact maths. Use it for every calculation.",
                   expression="the maths, e.g. 18% of 200000 or (45/60)*100"), "🧮 Calculator"),
    "get_weather": (get_weather, describe("get_weather", "Get the live weather right now for a city.",
                    city="city name, e.g. Jaipur"), "🌦️ Live weather"),
    "search_wikipedia": (search_wikipedia, describe("search_wikipedia", "Look up general knowledge (people, places, "
                         "history, science) on Wikipedia.", topic="a short topic, e.g. Hawa Mahal"), "📖 Wikipedia"),
    "ask_vgu_buddy": (ask_vgu_buddy, describe("ask_vgu_buddy", "Answer questions about Vivekananda Global University "
                      "(VGU), Jaipur: programmes, fees, hostels, clubs, events, placements, contact.",
                      question="the full question about VGU"), "🎓 VGU Buddy"),
    "add_reminder": (add_reminder, describe("add_reminder", "Save a reminder for the student.",
                     text="what to remember, with the date if given"), "📝 Save reminders (asks you first)"),
    "list_reminders": (list_reminders, describe("list_reminders", "Show all the student's saved reminders."),
                       "📋 Read reminders"),
}
AGENT_RULES = ("You are VGU Agent, a helpful assistant for students of VGU Jaipur. "
               "You cannot know today's date, the live weather, facts about VGU or the student's reminders by "
               "yourself, so always use the tools you have for them (add_reminder saves a reminder, list_reminders "
               "reads them). Never do maths in your head: use the calculator if you have it. "
               "If a tool returns an error, or you don't have the tool you need, say so; never invent a result. "
               "Answer in under 80 words.")


def run_agent(question, allowed, max_steps, show):  # the same loop as the notebook's 4.1
    messages = [{"role": "system", "content": AGENT_RULES}, {"role": "user", "content": question}]
    menu = [TOOLBOX[name][1] for name in allowed]
    steps = []
    for step in range(1, max_steps + 1):
        extra = {"tools": menu} if menu else {}             # all tools switched off → a plain chatbot
        reply = client.chat.completions.create(model=MODEL, messages=messages, temperature=0, **extra).choices[0].message
        if not reply.tool_calls:
            return reply.content or "", steps
        messages.append({"role": "assistant", "content": reply.content or "",
                         "tool_calls": [call.model_dump() for call in reply.tool_calls]})
        for call in reply.tool_calls:
            try:
                inputs = json.loads(call.function.arguments or "{}")
                result = str(TOOLBOX[call.function.name][0](**inputs)) if call.function.name in allowed \
                    else f"Error: the tool {call.function.name} is switched off."
            except Exception as error:
                inputs, result = {}, f"Error: {type(error).__name__}: {error}"
            for line in (f"**Step {step}** 🔧 `{call.function.name}({json.dumps(inputs, ensure_ascii=False)})`",
                         f"👀 {result[:300]}"):
                steps.append(line)
                show(line)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return "⛔ I stopped after too many steps without a final answer.", steps


# ---------- the page
st.set_page_config(page_title="VGU Agent", page_icon="🛠️")
st.title("🛠️ VGU Agent")
st.caption("An AI that picks its own tools: weather, calculator, Wikipedia, VGU Buddy, reminders · Project 3 · built by me!")
for key in ("messages", "pending"):
    st.session_state.setdefault(key, [])

with st.sidebar:
    st.subheader("Tools the agent may use")
    allowed = [name for name, (_, _, label) in TOOLBOX.items() if st.checkbox(label, value=True, key=name)]
    max_steps = st.slider("Maximum steps", 1, 8, 6)
    if st.button("🧹 New chat"):
        st.session_state.messages, st.session_state.pending = [], []
    st.caption(f"Model: {MODEL}")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m.get("steps"):
            with st.expander(f"🧠 How it worked: {len(m['steps']) // 2} tool call(s)"):
                st.markdown("\n\n".join(m["steps"]))
        st.markdown(m["content"])

if question := st.chat_input("Ask anything: weather, maths, VGU, Wikipedia, reminders..."):
    question = question.strip()[:1000]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.status("🤔 Thinking...", expanded=True) as status:
                answer, steps = run_agent(question, allowed, max_steps, show=st.markdown)
                status.update(label=f"🧠 How it worked: {len(steps) // 2} tool call(s)", state="complete", expanded=False)
        except Exception as error:                  # e.g. the AI server stopped
            st.error(f"The AI didn't answer ({type(error).__name__}). Check the setup cell, then try again.")
            st.stop()
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer, "steps": steps})

for i, text in enumerate(list(st.session_state.pending)):     # human in the loop: the student decides
    with st.container(border=True):
        st.markdown(f"🙋 **The agent wants to save a reminder:** {text}")
        yes, no = st.columns(2)
        if yes.button("✅ Save it", key=f"yes{i}"):
            with database() as db:
                db.execute("INSERT INTO reminders VALUES (?, ?)", (text, datetime.datetime.now().strftime("%d %b %H:%M")))
            st.session_state.pending.remove(text)
            st.session_state.messages.append({"role": "assistant", "content": f"✅ Saved your reminder: {text}"})
            st.rerun()
        if no.button("❌ Don't save", key=f"no{i}"):
            st.session_state.pending.remove(text)
            st.session_state.messages.append({"role": "assistant", "content": f"❌ OK, I didn't save: {text}"})
            st.rerun()
