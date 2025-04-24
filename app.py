import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

# App title and layout
st.set_page_config(page_title="SwimCoach AI Assistant", layout="wide")
st.title("🏊 SwimCoach AI Assistant")
st.markdown("Get personalized video recommendations for swimmer training using GenAI.")

# Sidebar – Athlete profile input
with st.sidebar:
    st.header("Athlete Profile")
    age = st.number_input("Age", min_value=8, max_value=25, value=14)
    skill = st.selectbox("Skill Level", ["Beginner", "Intermediate", "Advanced"])
    stroke = st.selectbox("Stroke", ["Freestyle", "Butterfly", "Backstroke", "Breaststroke"])
    goal = st.text_input("Training Goal", "Improve breathing rhythm")
    api_key = st.text_input("Enter your Groq API Key", type="password")

# File uploader for custom KB
st.sidebar.markdown("---")
kb_file = st.sidebar.file_uploader("Upload your KB CSV (Optional)", type=["csv"])

# Load KB and model
@st.cache_data
def load_kb(default=True):
    if default:
        return pd.read_csv("swimming_video_kb.csv")
    else:
        return pd.read_csv(kb_file)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# When user clicks the button
if st.button("Find Recommended Videos"):
    if not api_key:
        st.warning("Please enter your Groq API key.")
    else:
        try:
            os.environ["GROQ_API_KEY"] = api_key
            client = Groq(api_key=api_key)

            # Load KB and Model
            df = load_kb(default=(kb_file is None))
            model = load_model()

            # Create embeddings
            df["text"] = df["description"] + " " + df["transcript"]
            df["embedding"] = df["text"].apply(lambda x: model.encode(x).tolist())
            embeddings = np.array(df["embedding"].tolist()).astype("float32")

            # Build index and query
            index = build_index(embeddings)
            query = f"{stroke} {skill} swimmer, age {age}, wants to {goal}"
            query_vector = model.encode([query]).astype("float32")
            _, I = index.search(query_vector, k=3)
            top_entries = df.iloc[I[0]]
            context = "\n\n".join(top_entries["text"].tolist())

            # Prompt for Groq
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

            with st.spinner("Thinking like a coach..."):
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": "You are a helpful swimming coach assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                st.subheader("🏅 Recommended Videos")
                st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")