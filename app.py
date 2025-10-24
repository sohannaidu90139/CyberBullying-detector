import os
from pathlib import Path
import streamlit as st
import joblib

# Resolve file paths relative to this script
BASE_DIR = Path(__file__).parent

def local_css(file_path: Path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # optional: show a sidebar note if CSS not found (safe fallback)
        st.sidebar.warning(f"CSS file not found: {file_path.name}")

# Load CSS before rendering main UI
local_css(BASE_DIR / "designing.css")

# Load your trained model (use a safe path)
model_path = BASE_DIR / "cyber_model.pkl"
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.set_page_config(page_title="Cyberbullying Detection App", page_icon="🤖")

st.title("🤖 Cyberbullying Detection System")
st.markdown("This app detects whether a message contains harmful or bullying language.")

# Text input
text = st.text_area("Enter text to analyze:")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        prediction = model.predict([text])[0]
        proba = model.predict_proba([text])[0]
        confidence = max(proba) * 100

        if prediction == 1:
            st.error(f"🚨 Cyberbullying Detected! (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"✅ Safe message (Confidence: {confidence:.2f}%)")

st.sidebar.header("About Project")
st.sidebar.write("""
This AI-based tool analyzes online messages for bullying patterns
based on linguistic and behavioral features.

**Technologies Used:**  
- Python  
- scikit-learn  
- Streamlit
""")
