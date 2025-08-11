import streamlit as st
import os
import json
from pathlib import Path
from typing import List
import numpy as np
import faiss
from datetime import datetime
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ------------------------
# CONFIGURATION
# ------------------------

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    st.error("Please set TOGETHER_API_KEY environment variable before running.")
    st.stop()

# File paths
CLAIMS_FILE = r"claims_unstructured.txt"
CACHE_FILE = r"claim_embeddings.json"

# Together.ai LLM client
client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY
)

# Local embedding model (fast!)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------
# FUNCTIONS
# ------------------------
def load_claims(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data.strip().split("---\n")

def get_embedding_local(text: str) -> List[float]:
    """Local embedding generator for speed."""
    return embed_model.encode(text).tolist()

def build_faiss_index(embeddings: List[List[float]]) -> faiss.IndexFlatL2:
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

def search_claims(query: str, index: faiss.IndexFlatL2, claims: List[str], top_k: int = 5):
    query_emb = np.array(get_embedding_local(query)).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_emb, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(claims):
            results.append((claims[idx], dist))
    return results

def generate_answer(context_chunks: List[str], question: str) -> str:
    context_text = "\n\n---\n\n".join(context_chunks)
    prompt = f"""
You are an intelligent insurance claims assistant.
Use the provided claim documents below to answer the user's question.
If there is not enough information, say so clearly.

CONTEXT:
{context_text}

QUESTION:
{question}

Answer:
"""
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.2
    )
    return response.choices[0].message.content

# ------------------------
# LOAD DATA ONCE
# ------------------------
if "claims" not in st.session_state:
    st.session_state.claims = load_claims(CLAIMS_FILE)

if "faiss_index" not in st.session_state:
    # Load or compute embeddings
    if Path(CACHE_FILE).exists():
        claim_embeddings = json.loads(Path(CACHE_FILE).read_text())
    else:
        claim_embeddings = [get_embedding_local(c) for c in st.session_state.claims]
        Path(CACHE_FILE).write_text(json.dumps(claim_embeddings))
    st.session_state.faiss_index = build_faiss_index(claim_embeddings)

# ------------------------
# SESSION STATE FOR CHATS
# ------------------------
if "chats" not in st.session_state:
    st.session_state.chats = []
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(page_title="Boolean Chat", layout="wide")

# ------------------------
# SIDEBAR
# ------------------------
with st.sidebar:
    st.image(
        "https://booleandata.com/wp-content/uploads/2022/09/Boolean-logo_Boolean-logo-USA-1.png",
        use_container_width=True
    )

    # Footer icons
    # st.markdown(
    #     """
    #     <div style="position: fixed; bottom: 15px; display: flex; gap: 14px;">
    #         <a href="https://your-website.com" target="_blank">
    #             <img src="https://cdn-icons-png.flaticon.com/512/841/841364.png" width="22">
    #         </a>
    #         <a href="https://youtube.com/yourchannel" target="_blank">
    #             <img src="https://cdn-icons-png.flaticon.com/512/1384/1384060.png" width="22">
    #         </a>
    #         <a href="https://linkedin.com/in/yourprofile" target="_blank">
    #             <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="22">
    #         </a>
    #         <a href="mailto:your@email.com" target="_blank">
    #             <img src="https://cdn-icons-png.flaticon.com/512/561/561127.png" width="22">
    #         </a>
    #     </div>
    #     """, unsafe_allow_html=True
    # )

# ------------------------
# MAIN CHAT AREA
# ------------------------
st.title("💬AI-Powered Underwriting Assistant (Insurance) ")

if st.session_state.current_chat is None:
    current_messages = []
else:
    current_messages = st.session_state.chats[st.session_state.current_chat]["messages"]

for role, message in current_messages:
    bubble_color = "#DCF8C6" if role == "user" else "#F1F0F0"
    st.markdown(
        f"<div style='background-color:{bubble_color}; padding:10px; border-radius:10px; margin:5px 0; max-width:70%;'>{message}</div>",
        unsafe_allow_html=True
    )

# ------------------------
# FIXED BOTTOM INPUT BAR
# ------------------------
user_input = st.chat_input("Message Boolean Chat...")

if user_input:
    # If new chat
    if st.session_state.current_chat is None:
        chat_name = f"{user_input[:30]}..."
        st.session_state.chats.append({"name": chat_name, "messages": []})
        st.session_state.current_chat = len(st.session_state.chats) - 1

    # Append user message
    st.session_state.chats[st.session_state.current_chat]["messages"].append(("user", user_input))

    # 1. Search relevant claims (fast local embeddings)
    results = search_claims(user_input, st.session_state.faiss_index, st.session_state.claims, top_k=5)
    context_chunks = [r[0] for r in results]

    # 2. Generate AI answer
    bot_reply = generate_answer(context_chunks, user_input)

    # Append bot reply
    st.session_state.chats[st.session_state.current_chat]["messages"].append(("bot", bot_reply))

    st.rerun()
