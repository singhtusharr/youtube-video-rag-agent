"""Minimal Streamlit UI on top of the FastAPI backend.

Run the API first (`uvicorn main:app --reload`), then this
(`streamlit run streamlit_app.py`). Kept deliberately simple: a video
input, a chat-style question box, and the answer — enough to demo the
whole pipeline end to end.
"""
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="YouTube Video Assistant", page_icon="🎬")
st.title("🎬 YouTube Video Assistant")
st.caption("RAG + tool-calling agent over a YouTube video's transcript")

video_input = st.text_input("YouTube URL or video ID", placeholder="https://www.youtube.com/watch?v=...")

if "history" not in st.session_state:
    st.session_state.history = []

for role, message in st.session_state.history:
    with st.chat_message(role):
        st.write(message)

question = st.chat_input("Ask something about the video...")

if question:
    if not video_input:
        st.warning("Enter a YouTube URL or video ID first.")
    else:
        st.session_state.history.append(("user", question))
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={"video": video_input, "question": question},
                        timeout=120,
                    )
                    response.raise_for_status()
                    answer = response.json()["answer"]
                except requests.exceptions.RequestException as exc:
                    answer = f"Error calling API: {exc}"
                st.write(answer)
        st.session_state.history.append(("assistant", answer))
