import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from groq import Groq
import os
import chromadb
from chromadb.utils import embedding_functions

# Load model using transformers
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer, model

tokenizer, model = load_model()

def embed(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**inputs)
    embeddings = model_output.last_hidden_state.mean(dim=1)  # mean pooling
    return embeddings.numpy()

# Streamlit UI
st.set_page_config(page_title="SwimCoach AI Assistant", layout="wide")
st.title("🏊 SwimCoach AI Assistant")

# Sidebar
st.sidebar.header("Athlete Profile")
age = st.sidebar.number_input("Age", min_value=8, max_value=25, value=14)
skill = st.sidebar.selectbox("Skill Level", ["Beginner", "Intermediate", "Advanced"])
stroke = st.sidebar.selectbox("Stroke", ["Freestyle", "Butterfly", "Backstroke", "Breaststroke"])
goal = st.sidebar.text_input("Training Goal", "Improve breathing rhythm")
api_key = st.sidebar.text_input("Groq API Key", type="password")
kb_file = st.sidebar.file_uploader("Upload KB CSV", type=["csv"])

# Chroma setup
chroma_client = chromadb.Client()

if st.button("Get Recommendations"):
    if not api_key:
        st.warning("Please enter your Groq API key.")
    else:
        os.environ["GROQ_API_KEY"] = api_key
        client = Groq(api_key=api_key)

        if kb_file is not None:
            df = pd.read_csv(kb_file)
        else:
            st.error("Please upload a KB CSV file.")
            st.stop()

        df["text"] = df["description"] + " " + df["transcript"]
        texts = df["text"].tolist()
        ids = [f"id_{i}" for i in range(len(texts))]

        embeddings = embed(texts)
        collection = chroma_client.get_or_create_collection(name="swim_kb")

        # Clear and re-add to Chroma
        if len(collection.get(ids=[]).get("ids", [])) > 0:
            collection.delete(ids=collection.get()["ids"])
        collection.add(documents=texts, ids=ids, embeddings=embeddings.tolist())

        # Query embedding
        athlete_query = f"{stroke} {skill} swimmer, age {age}, wants to {goal}"
        query_embedding = embed([athlete_query])[0]
        result = collection.query(query_embeddings=[query_embedding], n_results=3)
        context = "\n\n".join(result["documents"][0])

        # Prompt
        prompt = f"""
        You are a professional swimming coach assistant.

        Athlete Profile:
        - Age: {age}
        - Skill Level: {skill}
        - Stroke: {stroke}
        - Goal: {goal}

        Based on the following training video content, suggest 2–3 YouTube videos that are likely to help.
        For each:
        1. Title (based on content)
        2. Why it's relevant
        3. Expected improvements

        Context:
        {context}
        """

        with st.spinner("Generating your coaching insights..."):
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful swimming coach assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            st.subheader("🏅 Recommended Videos")
            st.write(response.choices[0].message.content)