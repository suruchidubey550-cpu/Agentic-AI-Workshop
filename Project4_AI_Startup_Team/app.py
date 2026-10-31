# 5.1 AI Startup Team: a team of AI agents turns a one-line idea into a pitch and a poster
import io, json, os
import requests, torch
import streamlit as st
from diffusers import AutoPipelineForText2Image
from openai import OpenAI
from PIL import ImageDraw, ImageFont

client = OpenAI(base_url=os.environ["VGU_BASE_URL"], api_key=os.environ["VGU_API_KEY"])
MODEL = os.environ["VGU_MODEL"]
HEADERS = {"User-Agent": "VGU-GenAI-Workshop/1.0 (student project)"}
BLOCKED = ["nude", "naked", "sexy", "porn", "blood", "gore", "kill", "gun", "weapon", "drug", "alcohol", "cigarette"]

# ---------- the team's job descriptions (the same as the notebook's Parts 2-4)
PLANNER = ("You are the Planner of a startup team. Turn the idea into a short plan with three parts: "
           "Problem, Users, and 3 Key features. Use bullet points, under 90 words.")
RESEARCHER = ("You are the Researcher of a startup team. Using ONLY the facts given, write 2 short bullet points "
              "that make this startup's pitch stronger.")
WRITER = ('You are the Writer of a startup team. Reply in JSON with exactly these keys: "name" (a catchy 1-2 word '
          'startup name), "tagline" (at most 8 words), "pitch" (3 sentences: problem, solution, why now), '
          '"insta_post" (under 40 words, with 3 hashtags).')
CRITIC = ('You are a strict startup investor on a TV show. Judge the pitch honestly; most first pitches score '
          'between 4 and 7. Reply in JSON with exactly these keys: "score" (a whole number from 1 to 10), '
          '"good" (one sentence), "improve" (one concrete fix the Writer should make).')
DESIGNER = ("Describe a striking, family-friendly poster picture for this startup in under 40 words. "
            "Bright colours, digital art. No text, letters or logos in the picture.")


def ask(task, role, temperature=0.7):
    messages = [{"role": "system", "content": role}, {"role": "user", "content": task}]
    return client.chat.completions.create(model=MODEL, messages=messages, temperature=temperature).choices[0].message.content


def ask_json(task, role, temperature=0.7):
    messages = [{"role": "system", "content": role}, {"role": "user", "content": task}]
    response = client.chat.completions.create(model=MODEL, messages=messages, temperature=temperature,
                                              response_format={"type": "json_object"})
    return json.loads(response.choices[0].message.content)


def score_of(review):                                # "7", 7 or "7/10" → 7
    try:
        return int(str(review.get("score", 0)).split("/")[0])
    except ValueError:
        return 0


def search_wikipedia(topic):
    hits = requests.get("https://en.wikipedia.org/w/api.php", headers=HEADERS, timeout=15, params={
        "action": "query", "list": "search", "srsearch": topic, "srlimit": 1, "format": "json"}).json()["query"]["search"]
    if not hits:
        return f"No Wikipedia article found for {topic!r}."
    page = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/" + hits[0]["title"].replace(" ", "_"),
                        headers=HEADERS, timeout=15).json()
    return f"{hits[0]['title']}: {page.get('extract', '')[:600]}"


# ---------- the painter and the lettering (the same as the notebook's 0.2 and 4.2)
@st.cache_resource
def load_painter():
    if torch.cuda.is_available():
        painter = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", dtype=torch.float16,
                                                            variant="fp16", requires_safety_checker=False)
        if torch.cuda.get_device_properties(0).total_memory > 12e9:
            painter.to("cuda")
        else:
            painter.enable_model_cpu_offload()
        return painter
    return AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", requires_safety_checker=False)


def font(size):
    for path in ["/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/liberation/LiberationSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def write_centered(draw, y, text, size, colour):
    while size > 10 and draw.textlength(text, font=font(size)) > 490:
        size -= 1
    draw.text((256, y), text, font=font(size), fill=colour, anchor="mm")


def make_poster(pitch, scene):
    picture = load_painter()(prompt=scene, num_inference_steps=2, guidance_scale=0.0, width=512, height=512).images[0]
    draw = ImageDraw.Draw(picture)
    draw.rectangle([0, 410, 512, 512], fill=(20, 20, 45))
    write_centered(draw, 445, str(pitch.get("name", "")).upper(), 40, "white")
    write_centered(draw, 487, str(pitch.get("tagline", "")), 22, (255, 210, 90))
    return picture


# ---------- the page
st.set_page_config(page_title="AI Startup Team", page_icon="🏢")
st.title("🏢 AI Startup Team")
st.caption("A team of AI agents (one model, different jobs) turns your idea into a pitch and a poster · Project 4 · built by me!")

with st.sidebar:
    st.subheader("The team")
    use_researcher = st.checkbox("🔎 Researcher (reads Wikipedia)", value=True)
    max_rounds = st.slider("🔁 Rounds of improvement (at most)", 0, 3, 2)
    pass_mark = st.slider("🧐 Score the Critic must give", 5, 10, 8)
    use_designer = st.checkbox("🎨 Designer (makes the poster)", value=True)
    st.caption(f"Model: {MODEL}")

idea = st.text_input("💡 Your startup idea, in one line", placeholder="An app that helps VGU students find lost items on campus")
if st.button("🚀 Start the team", type="primary"):
    idea = idea.strip()[:300]
    if not idea:
        st.warning("Type your idea first. ✏️")
        st.stop()
    if any(word in idea.lower() for word in BLOCKED):                 # our own safety filter
        st.error("Let's keep it college-friendly: please try another idea.")
        st.stop()
    team, rounds, poster = {"idea": idea}, 0, None
    try:
        with st.status("🧭 The Planner is planning...", expanded=True) as status:
            team["plan"] = ask(idea, PLANNER, temperature=0.5)
            st.markdown("**🧭 Planner**\n\n" + team["plan"])
            if use_researcher:
                status.update(label="🔎 The Researcher is reading Wikipedia...")
                topic = ask(idea, "Give a 1-3 word Wikipedia topic about this idea's market or technology. "
                            "Reply with the topic only.", temperature=0).strip().strip('".')
                try:
                    facts = search_wikipedia(topic)
                except Exception:
                    facts = "Wikipedia could not be reached."
                team["research"] = ask(f"Idea: {idea}\nFacts about {topic}: {facts}", RESEARCHER, temperature=0.3)
                st.markdown(f"**🔎 Researcher** (read *{topic}* on Wikipedia)\n\n" + team["research"])
            status.update(label="✍️ The Writer is writing the pitch...")
            notes = "\n".join(f"{key}: {value}" for key, value in team.items())
            team["pitch"] = ask_json(notes, WRITER, temperature=0.9)
            st.markdown(f"**✍️ Writer:** {team['pitch'].get('name')} · *{team['pitch'].get('tagline')}*")
            status.update(label="🧐 The Critic is judging...")
            review = ask_json(json.dumps(team["pitch"]), CRITIC, temperature=0.3)
            st.markdown(f"**🧐 Critic:** {score_of(review)}/10 · {review.get('improve', '')}")
            while score_of(review) < pass_mark and rounds < max_rounds:
                rounds += 1
                status.update(label=f"🔁 Round {rounds}: the Writer is improving the pitch...")
                task = (notes + f"\nYour last draft: {json.dumps(team['pitch'])}"
                              + f"\nThe investor says: {review.get('improve', '')}\nWrite a better version.")
                team["pitch"] = ask_json(task, WRITER, temperature=0.9)
                review = ask_json(json.dumps(team["pitch"]), CRITIC, temperature=0.3)
                st.markdown(f"**🔁 Round {rounds}:** {team['pitch'].get('name')} · *{team['pitch'].get('tagline')}* "
                            f"→ 🧐 {score_of(review)}/10")
            if use_designer:
                status.update(label="🎨 The Designer is making the poster...")
                scene = ask(f"{team['pitch'].get('name')}: {team['pitch'].get('tagline')}. {team['pitch'].get('pitch')}",
                            DESIGNER, temperature=0.9)
                if any(word in scene.lower() for word in BLOCKED):
                    scene = "a bright, friendly digital illustration of happy college students using a new app on campus"
                poster = make_poster(team["pitch"], scene)
                st.markdown("**🎨 Designer:** " + scene)
            status.update(label=f"✅ Done: the team finished, with {rounds} round(s) of improvement",
                          state="complete", expanded=False)
    except Exception as error:                      # e.g. bad JSON from the AI, or the server stopped
        st.error(f"The team got stuck ({type(error).__name__}). Click Start again.")
        st.stop()
    st.session_state.result = (team, review, rounds, poster)

if "result" in st.session_state:                    # the final pitch page (kept when you click a download button)
    team, review, rounds, poster = st.session_state.result
    pitch = team["pitch"]
    st.header(f"🚀 {pitch.get('name', '')}")
    st.subheader(pitch.get("tagline", ""))
    left, right = st.columns([3, 2])
    left.markdown(pitch.get("pitch", ""))
    left.info("📸 " + str(pitch.get("insta_post", "")))
    left.metric("🧐 Critic's score", f"{score_of(review)}/10", f"after {rounds} round(s) of improvement", delta_color="off")
    page = (f"# {pitch.get('name')}\n*{pitch.get('tagline')}*\n\n{pitch.get('pitch')}\n\n## Plan\n{team['plan']}\n\n"
            f"## Instagram post\n{pitch.get('insta_post')}\n\n## Critic's review: {score_of(review)}/10\n"
            f"- 👍 {review.get('good', '')}\n- 🔧 {review.get('improve', '')}\n")
    left.download_button("⬇️ Download the pitch (.md)", page, file_name="pitch.md")
    if poster is not None:
        right.image(poster)
        png = io.BytesIO()
        poster.save(png, format="PNG")
        right.download_button("⬇️ Download the poster", png.getvalue(), file_name="poster.png", mime="image/png")
