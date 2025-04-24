import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
from groq import Groq
import os
import chromadb
from chromadb.utils import embedding_functions

# Configure Chroma client
chroma_client = chromadb.Client()
chroma_collection = None

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Sidebar inputs
st.sidebar.header("Athlete Profile")
age = st.sidebar.number_input("Age", min_value=8, max_value=25, value=14)
skill = st.sidebar.selectbox("Skill Level", ["Beginner", "Intermediate", "Advanced"])
stroke = st.sidebar.selectbox("Stroke", ["Freestyle", "Butterfly", "Backstroke", "Breaststroke"])
goal = st.sidebar.text_input("Training Goal", "Improve breathing rhythm")
api_key = st.sidebar.text_input("Groq API Key", type="password")
kb_file = st.sidebar.file_uploader("Upload KB CSV", type=["csv"])

st.title("🏊 SwimCoach AI Assistant")

if st.button("Get Recommendations"):
    if not api_key:
        st.warning("Please enter your Groq API key.")
    else:
        os.environ["GROQ_API_KEY"] = api_key
        client = Groq(api_key=api_key)

        # Load KB
        if kb_file is not None:
            df = pd.read_csv(kb_file)
        else:
            st.error("Please upload a KB CSV file.")
            st.stop()

        df["text"] = df["description"] + " " + df["transcript"]
        texts = df["text"].tolist()
        ids = [f"id_{i}" for i in range(len(texts))]

        # Embed and store with Chroma
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        chroma_collection = chroma_client.get_or_create_collection(name="swim_kb", embedding_function=embedding_fn)
        chroma_collection.add(documents=texts, ids=ids)

        # Query
        athlete_query = f"{stroke} {skill} swimmer, age {age}, wants to {goal}"
        results = chroma_collection.query(query_texts=[athlete_query], n_results=3)
        context = "\n\n".join(results["documents"][0])

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

        with st.spinner("Analyzing profiles..."):
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful swimming coach assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            st.subheader("🏅 Recommended Videos")
            st.write(response.choices[0].message.content)