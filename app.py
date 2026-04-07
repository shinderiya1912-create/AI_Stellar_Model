import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import base64
import os
import requests
import tensorflow as tf
import plotly.graph_objects as go
import speech_recognition as sr
import requests
import matplotlib.pyplot as plt
from datetime import datetime   # ⭐ ADDED

# -------------------------------
# PAGE CONFIG (MUST BE FIRST)
# -------------------------------
st.set_page_config(page_title="Stellar AI System", layout="wide")

# -------------------------------
# LOAD MODELS
# -------------------------------
def load_file(file):
    if os.path.exists(file):
        return joblib.load(file)
    else:
        st.error(f"❌ Missing file: {file}")
        st.stop()

model = load_file("stellar_rf_model.pkl")
scaler = load_file("scaler.pkl")
le = load_file("label_encoder.pkl")

import shap

# Create SHAP explainer once
explainer = shap.TreeExplainer(model)

# CNN SAFE LOAD
cnn_model = None
try:
    if os.path.exists("cnn_model.h5"):
        cnn_model = tf.keras.models.load_model("cnn_model.h5")
except Exception as e:
    st.warning(f"CNN load failed: {e}")

# -------------------------------
# BACKGROUND
# -------------------------------
def set_bg_image(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,20,0.9), rgba(0,0,20,0.95)),
                        url("data:image/png;base64,{encoded}");
            background-size: cover;
            color: white;
        }}
        h1, h2, h3 {{
            text-shadow: 0px 0px 10px cyan;
        }}
        </style>
        """, unsafe_allow_html=True)

set_bg_image("space_background.png")

# -------------------------------
# TITLE
# -------------------------------
st.title("🌌 Stellar AI System")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("🛰️ Control Panel")

option = st.sidebar.radio(
    "Choose Mode",
    ["Manual Input", "Upload Image", "Chatbot", "Explore More"]
)

if option == "Explore More":
    st.switch_page("pages/explore_more.py")

# -------------------------------
# NASA APOD  ⭐ UPDATED FUNCTION
# -------------------------------
def get_nasa_apod():

    try:
        url = "https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY"

        r = requests.get(url, timeout=5)

        if r.status_code == 200:

            data = r.json()

            # ⭐ GET CURRENT DATE & TIME
            now = datetime.now()

            today_date = now.strftime("%Y-%m-%d")
            current_time = now.strftime("%H:%M:%S")

            # ⭐ PREPARE RECORD
            record = {
                "Date": today_date,
                "Time": current_time,
                "Title": data.get("title", ""),
                "Image_URL": data.get("url", "")
            }

            file_name = "nasa_apod_log.csv"

            df_new = pd.DataFrame([record])

            # ⭐ SAVE TO CSV
            if os.path.exists(file_name):

                df_old = pd.read_csv(file_name)

                df_all = pd.concat(
                    [df_old, df_new],
                    ignore_index=True
                )

            else:

                df_all = df_new

            df_all.to_csv(file_name, index=False)

            return data

    except:
        return None

# -------------------------------
# VOICE INPUT
# -------------------------------
def voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Speak now...")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        return text
    except:
        return "Voice not recognized"

# -------------------------------
# MANUAL INPUT WITH SHAP
# -------------------------------
if option == "Manual Input":

    st.subheader("🔢 Enter Features")

    try:
        features = list(scaler.feature_names_in_)
    except:
        st.error("Scaler missing feature names.")
        st.stop()

    input_data = []

    for f in features:

        val = st.number_input(
            f,
            value=0.0,
            format="%.6f"
        )

        input_data.append(val)

    if st.button("Predict"):

        df = pd.DataFrame(
            [input_data],
            columns=features
        )

        scaled = scaler.transform(df)

        pred = model.predict(scaled)

        result = le.inverse_transform(pred)[0]

        st.success(f"🌟 Predicted Class: {result}")

        st.subheader("🔍 Why this prediction? (SHAP Explanation)")

        try:

            shap_values = explainer.shap_values(scaled)

            shap_df = pd.DataFrame(
                shap_values[0],
                columns=features
            )

            fig, ax = plt.subplots(figsize=(10,6))

            shap_df.T.plot.barh(
                ax=ax
            )

            ax.set_title("Feature Contribution to Prediction")

            st.pyplot(fig)

        except Exception as e:

            st.warning("⚠️ SHAP explanation not available.")
            st.write(e)

# -------------------------------
# IMAGE UPLOAD + CNN
# -------------------------------
if option == "Upload Image":

    st.subheader("🖼️ Upload Space Image")

    img_file = st.file_uploader(
        "Upload Image",
        type=["jpg","png","jpeg"]
    )

    if img_file:

        image = Image.open(img_file)

        st.image(image)

        if cnn_model:

            img = image.resize((128,128))

            img = np.array(img)/255.0

            img = np.expand_dims(img, axis=0)

            pred = cnn_model.predict(img)

            class_names = ['GALAXY','STAR','QSO']

            result = class_names[np.argmax(pred)]

            confidence = np.max(pred)

            st.success(f"🌌 {result}")

            st.progress(float(confidence))

# -------------------------------
# REAL-TIME SPACE INFO
# -------------------------------
def get_space_info(query):

    try:

        query = query.strip().replace(" ", "_")

        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"

        response = requests.get(url, timeout=5)

        if response.status_code == 200:

            data = response.json()

            if "extract" in data:
                return data["extract"]

        return "❌ I couldn't find detailed information on that topic."

    except:
        return "⚠️ Error retrieving information."

# -------------------------------
# CHATBOT
# -------------------------------
if option == "Chatbot":

    st.subheader("🤖 Space AI Assistant")

    st.markdown(
        "Ask anything about **space, stars, galaxies, planets, black holes, missions**, and more."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_input(
        "💬 Ask something about space:"
    )

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []

    if st.button("🎙️ Voice Input"):

        voice_text = voice_input()

        if voice_text:

            st.write("🎤 You said:", voice_text)

            user_input = voice_text

    if user_input:

        st.session_state.messages.append(
            ("You", user_input)
        )

        response = get_space_info(user_input)

        st.session_state.messages.append(
            ("AI", response)
        )

    st.markdown("### 💬 Conversation")

    for role, msg in st.session_state.messages:

        if role == "You":
            st.markdown(f"🧑 **You:** {msg}")
        else:
            st.markdown(f"🤖 **AI:** {msg}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("🚀 Built with AI + Streamlit")