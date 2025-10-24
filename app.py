import streamlit as st
import joblib

# Load your trained model
model = joblib.load('cyber_model.pkl')

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
