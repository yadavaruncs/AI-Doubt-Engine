"""Minimal Streamlit front end. Shows sources, route and latency — the three
things a reviewer looks for to tell a real RAG app from a wrapper."""

import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Doubt Engine", page_icon="🔬", layout="centered")
st.title("🔬 AI Doubt Engine")
st.caption("Hybrid retrieval + reranking + tool-using agent over your physics corpus.")

with st.sidebar:
    st.subheader("Session")
    use_cache = st.toggle("Semantic cache", value=True)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    try:
        health = requests.get(f"{API}/healthz", timeout=5).json()
        st.metric("Indexed chunks", health.get("points", "—"))
        m = requests.get(f"{API}/metrics", timeout=5).json()
        st.metric("p95 latency (ms)", m["latency_ms"]["p95"])
        st.metric("Cache hit rate", m["cache"]["hit_rate"])
    except Exception:
        st.warning("Backend unreachable.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            meta = msg["meta"]
            bits = [f"`{meta.get('route')}`", f"{meta.get('latency_ms')} ms"]
            if meta.get("cached"):
                bits.append(f"cached (sim {meta.get('cache_similarity')})")
            st.caption(" · ".join(bits))
            if meta.get("sources"):
                with st.expander("Sources"):
                    for s in meta["sources"]:
                        st.write(
                            f"**[{s['n']}]** {s['source']}"
                            + (f", p.{s['page']}" if s.get("page") else "")
                            + f" — score {s['score']}"
                        )
            if meta.get("tool_calls"):
                with st.expander("Tool calls"):
                    st.json(meta["tool_calls"])

if prompt := st.chat_input("Ask a physics doubt..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning..."):
            try:
                r = requests.post(
                    f"{API}/ask",
                    json={"question": prompt, "use_cache": use_cache},
                    timeout=120,
                )
                r.raise_for_status()
                data = r.json()
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                st.stop()
        st.markdown(data["answer"])
        st.session_state.messages.append(
            {"role": "assistant", "content": data["answer"], "meta": data}
        )
        st.rerun()
