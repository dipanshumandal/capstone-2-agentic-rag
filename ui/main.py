"""
Streamlit chat UI for the support agent. Talks to the FastAPI backend.
run:  streamlit run ui/main.py
"""
import os
import uuid
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")

st.set_page_config(page_title="ShopEase Support", page_icon="🛍️", layout="centered")

# whatsapp style bubbles
st.markdown("""
<style>
.bubble {padding:10px 14px; border-radius:12px; margin:6px 0; max-width:80%; font-size:15px; color:#111;
         box-shadow:0 1px 1px rgba(0,0,0,.15); white-space:pre-wrap;}
.user {background:#dcf8c6; margin-left:auto; border-bottom-right-radius:2px;}
.bot {background:#ffffff; margin-right:auto; border-bottom-left-radius:2px;}
.meta {font-size:11px; color:#666; margin-top:4px;}
</style>
""", unsafe_allow_html=True)

st.title("🛍️ ShopEase Support")
st.caption("Agentic RAG assistant - LangGraph + Gemini + FAISS")

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]
if "history" not in st.session_state:
    st.session_state.history = []     # list of dicts: role, text, meta

with st.sidebar:
    st.subheader("Session")
    st.code(st.session_state.session_id)
    if st.button("New conversation"):
        st.session_state.session_id = uuid.uuid4().hex[:8]
        st.session_state.history = []
        st.rerun()
    try:
        h = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"API ok - {h['chunks']} chunks, llm: {h['llm']}")
    except Exception:
        st.error(f"API not reachable at {API_URL}")
    show_context = st.checkbox("Show retrieved context", value=True)


def send_feedback(item, rating, correction=None):
    try:
        requests.post(f"{API_URL}/feedback", json={
            "session_id": st.session_state.session_id,
            "query": item["query"],
            "answer": item["text"],
            "rating": rating,
            "correction": correction or None,
            "intent": item["meta"].get("intent"),
        }, timeout=10)
        item["rated"] = rating
    except Exception as e:
        st.warning(f"could not send feedback: {e}")


# ---- chat history ----
for i, item in enumerate(st.session_state.history):
    if item["role"] == "user":
        st.markdown(f'<div class="bubble user">{item["text"]}</div>', unsafe_allow_html=True)
    else:
        m = item["meta"]
        esc = f" · escalated ({m.get('ticket_id')})" if m.get("escalated") else ""
        st.markdown(
            f'<div class="bubble bot">{item["text"]}'
            f'<div class="meta">intent: {m.get("intent")} · sentiment: {m.get("sentiment", {}).get("sentiment")} '
            f'· confidence: {m.get("confidence")}{esc}</div></div>',
            unsafe_allow_html=True)
        if show_context:
            with st.expander("retrieved context + agent trace"):
                st.write("agents:", " → ".join(m.get("agents_invoked", [])))
                st.write("sources:", m.get("sources"))
                st.write("validation:", m.get("validation"))
                for j, c in enumerate(m.get("retrieved_context", []), 1):
                    st.text(f"[{j}] {c[:400]}")
        c1, c2, c3 = st.columns([1, 1, 6])
        if item.get("rated"):
            c1.caption("rated 👍" if item["rated"] == "up" else "rated 👎")
        else:
            if c1.button("👍", key=f"up{i}"):
                send_feedback(item, "up")
                st.rerun()
            if c2.button("👎", key=f"down{i}"):
                send_feedback(item, "down")
                st.rerun()
        if item.get("rated") == "down":
            corr = c3.text_input("what should the answer have been?", key=f"corr{i}")
            if corr and c3.button("send correction", key=f"sc{i}"):
                send_feedback(item, "down", corr)
                st.success("correction saved")

# ---- input ----
user_text = st.chat_input("Type your message...")
if user_text:
    st.session_state.history.append({"role": "user", "text": user_text})
    with st.spinner("agents working..."):
        try:
            r = requests.post(f"{API_URL}/query",
                              json={"session_id": st.session_state.session_id, "query": user_text}, timeout=120)
            r.raise_for_status()
            data = r.json()
            st.session_state.history.append({"role": "bot", "text": data["answer"], "meta": data, "query": user_text})
        except Exception as e:
            st.session_state.history.append({"role": "bot", "text": f"Sorry, something went wrong: {e}",
                                             "meta": {}, "query": user_text})
    st.rerun()
