import streamlit as st
import joblib

# Load model + vectorizer
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# ----- PAGE CONFIG -----
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ----- CUSTOM DARK THEME CSS -----
st.markdown("""
    <style>
        body {
            background-color: #000000;
        }
        .main {
            background-color: red;
        }
        h1, h2, h3, p, label {
            color: #000000 !important;
        }
        .stTextArea textarea {
            background-color: #161b22 !important;
            color: #ffffff !important;
            border-radius: 10px;
            border: 1px solid #30363d;
        }
        .stButton>button {
            background-color: #238636 !important;
            color: white !important;
            border-radius: 8px;
            height: 3em;
            width: 12em;
            border: none;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #2ea043 !important;
            border: none;
        }
        .result-box {
            padding: 15px;
            border-radius: 10px;
            color: white;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ----- HEADER -----
st.markdown("<h1 style='text-align:center;'>📰 Fake News Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter a news article below to check if it is REAL or FAKE.</p>", unsafe_allow_html=True)

# ----- INPUT -----
news_input = st.text_area("Enter News Text", height=200)

# ----- BUTTON -----
if st.button("Analyze News"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("✔️ This news article is **REAL**.")
        else:
            st.error("❌ This news article is **FAKE**.")
