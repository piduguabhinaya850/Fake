import streamlit as st
import pickle

st.title("📰 Fake News Detection System")

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

text = st.text_area("Paste news text here")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]

        if pred == 0:
            st.error("❌ FAKE NEWS")
        else:
            st.success("✅ REAL NEWS")