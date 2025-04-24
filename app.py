import streamlit as st
import pandas as pd
import os
from groq import Groq

# UI
st.set_page_config(page_title="SwimCoach AI Assistant", layout="wide")
st.title("🏊 SwimCoach AI Assistant")

# Sidebar
with st.sidebar:
    st.header("Athlete Profile")
    age = st.number_input("Age", min_value=8, max_value=25, value=14)
    skill = st.selectbox("Skill Level", ["Beginner", "Intermediate", "Advanced"])
    stroke = st.selectbox("Stroke", ["Freestyle", "Butterfly", "Backstroke", "Breaststroke"])
    goal = st.text_input("Training Goal", "Improve breathing rhythm")
    api_key = st.text_input("Groq API Key", type="password")
    kb_file = st.file_uploader("Upload KB CSV", type=["csv"])

# Submit button
if st.button("Get Recommendations"):
    if not api_key:
        st.warning("Please enter your Groq API key.")
        st.stop()

    os.environ["GROQ_API_KEY"] = api_key
    client = Groq(api_key=api_key)

    if kb_file is None:
        st.error("Please upload your KB CSV.")
        st.stop()

    # Load KB
    df = pd.read_csv(kb_file)
    df["text"] = df["description"] + " " + df["transcript"]

    # Filter relevant videos using basic keyword match
    query_keywords = f"{stroke} {skill} {goal}".lower()
    filtered_df = df[df["text"].str.lower().str.contains(stroke.lower()) | 
                     df["text"].str.lower().str.contains(skill.lower()) |
                     df["text"].str.lower().str.contains(goal.lower())]

    top_context = "\n\n".join(filtered_df["text"].head(3).tolist())

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
    {top_context}
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
