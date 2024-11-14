import streamlit as st
import pickle
from streamlit_lottie import st_lottie
import numpy as np

# Load model and scaler from pickle files
def load_model():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

model, scaler = load_model()

# Function to load Lottie animations
def load_lottie(url: str):
    import requests
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

# Load Lottie animation
lottie_url = "https://assets4.lottiefiles.com/packages/lf20_jcikwtux.json"  # Lottie animation URL
lottie_animation = load_lottie(lottie_url)

# Custom CSS styling
st.markdown(
    """
    <style>
        .main {
            background-color: #f5f5f5;
            font-family: Arial, sans-serif;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .prediction-output {
            background-color: #e8f5e9;
            color: #1b5e20;
            font-weight: bold;
            padding: 15px;
            border-radius: 10px;
            font-size: 18px;
            text-align: center;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with Lottie animation
with st.sidebar:
    st.header("🧠 Sentiment Analysis")
    st.write("Analyze the sentiment of your text input.")
    st_lottie(lottie_animation, height=150, key="animation")

# App title and text input
st.title("Sentiment Analysis App")
st.write("Enter text below to analyze its sentiment.")

# Text area for input
user_input = st.text_area("Type your text here:")

# Predict button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            # Transform input text to model format
            # Assume the model expects scaled text features (e.g., TF-IDF or embeddings)
            transformed_input = scaler.transform([user_input])
            prediction = model.predict(transformed_input)
            
            # Display prediction result
            sentiment = "Positive 🤗" if prediction[0] == 1 else "Negative 🥺 "
            st.markdown(
                f"<div class='prediction-output'>The sentiment is: {sentiment}</div>",
                unsafe_allow_html=True
            )
    else:
        st.error("Please enter some text to analyze.")

# Footer with custom styling
st.markdown(
    """
    <div style='text-align: center; padding-top: 20px;'>
        <small>Built with ❤️ using Streamlit</small>
    </div>
    """,
    unsafe_allow_html=True
)
