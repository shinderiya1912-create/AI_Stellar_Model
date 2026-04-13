import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import base64
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import shap
import json

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="Stellar AI System",
    layout="wide"
)

# -------------------------------
# LOAD MODELS
# -------------------------------

def load_file(file):

    if os.path.exists(file):
        return joblib.load(file)

    else:
        st.error(f"❌ Missing file: {file}")
        st.stop()


model = load_file("models/stellar_rf_model.pkl")
scaler = load_file("models/scaler.pkl")
le = load_file("models/label_encoder.pkl")
explainer = shap.TreeExplainer(model)

# -------------------------------
# LOAD CNN MODEL
# -------------------------------

cnn_model = None

try:

    if os.path.exists("models/cnn_model.h5"):

        cnn_model = tf.keras.models.load_model(
            "models/cnn_model.h5"
        )

except Exception as e:

    st.warning(f"CNN load failed: {e}")

# -------------------------------
# BACKGROUND IMAGE
# -------------------------------

def set_bg_image(image_file):

    if os.path.exists(image_file):

        with open(image_file, "rb") as f:

            encoded = base64.b64encode(
                f.read()
            ).decode()

        st.markdown(f"""
        <style>
        .stApp {{
            background:
            linear-gradient(
                rgba(0,0,20,0.9),
                rgba(0,0,20,0.95)
            ),
            url("data:image/png;base64,{encoded}");
            background-size: cover;
        }}
        </style>
        """, unsafe_allow_html=True)

set_bg_image("assets/space_background.png")

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

    [
        "Manual Input",
        "Upload Image",
    ]
)

# -------------------------------
# OBJECT INFORMATION
# -------------------------------

object_info = {

"galaxy":
"A galaxy is a massive system of billions of stars, gas, dust, and dark matter bound together by gravity.",

"black_hole":
"A black hole is a region in space where gravity is extremely strong.",

"mars":
"Mars is the fourth planet from the Sun and is known as the Red Planet.",

"jupiter":
"Jupiter is the largest planet in the solar system.",

"saturn":
"Saturn is famous for its large ring system.",

"earth":
"Earth is the third planet from the Sun and supports life.",

"venus":
"Venus has extremely high surface temperatures.",

"neptune":
"Neptune is a distant ice giant.",

"uranus":
"Uranus rotates on its side.",

"mercury":
"Mercury is closest to the Sun.",

"asteroid":
"A small rocky object orbiting the Sun.",

"nebula":
"A large cloud of gas and dust in space."

}

# -------------------------------
# FEATURE MEANINGS
# -------------------------------

feature_meaning = {

"u": "Ultraviolet band magnitude",
"g": "Green band magnitude",
"r": "Red band magnitude",
"i": "Near-infrared band magnitude",
"z": "Redshift — indicates distance",
"redshift": "Velocity and distance indicator",
"ra": "Right Ascension",
"dec": "Declination",
"cam_col": "Camera column used",
"field_ID": "Field number",
"run_ID": "Observation run ID",
"fiber_ID": "Fiber ID",
"spec_obj_ID": "Spectral object ID"

}

# -------------------------------
# RECOMMENDATIONS
# -------------------------------

recommendations = {

"GALAXY":
"This object is likely a galaxy. Further observation recommended.",

"STAR":
"This object appears to be a star.",

"QSO":
"This object may be a quasar candidate."

}

# -------------------------------
# MANUAL INPUT
# -------------------------------

if option == "Manual Input":

    st.subheader("🔢 Enter Features")

    features = list(
        scaler.feature_names_in_
    )

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

        proba = model.predict_proba(scaled)

        confidence = np.max(proba) * 100

        st.success(
            f"🌟 Predicted Class: {result}"
        )

        st.write(
            f"📊 Confidence: {confidence:.2f}%"
        )

        # Reliability

        if confidence > 85:

            st.success("✅ High confidence prediction")

        elif confidence > 60:

            st.warning("⚠️ Moderate confidence prediction")

        else:

            st.error("❌ Low confidence prediction")

        # Probability Chart

        prob_df = pd.DataFrame({

            "Class": le.classes_,
            "Probability": proba[0]

        })

        st.subheader(
            "📊 Class Probability Distribution"
        )

        st.bar_chart(
            prob_df.set_index("Class")
        )

        # Input Summary

        st.subheader("📋 Input Summary")

        st.dataframe(df)

        # -------------------------------
        # SHAP
        # -------------------------------

        st.subheader(
            "🧠 Feature Contribution Analysis"
        )

        try:

            shap_values = explainer.shap_values(
                scaled
            )

            shap_array = np.array(
                shap_values
            ).reshape(-1)[:len(features)]

            shap_df = pd.DataFrame({

                "Feature": features,
                "Contribution": shap_array

            })

            shap_df = shap_df.sort_values(

                by="Contribution",
                key=abs,
                ascending=False

            )

            fig, ax = plt.subplots()

            ax.barh(
                shap_df["Feature"],
                shap_df["Contribution"]
            )

            ax.invert_yaxis()

            st.pyplot(fig)

            # Top Features

            st.subheader(
                "📖 Top Influencing Features"
            )

            top_features = shap_df.head(5)

            for _, row in top_features.iterrows():

                direction = (

                    "increased"
                    if row["Contribution"] > 0
                    else "decreased"

                )

                st.write(

                    f"• **{row['Feature']}** "
                    f"{direction} prediction confidence."

                )

            # Feature Meaning

            st.subheader("📖 Feature Meaning")

            for f in top_features["Feature"]:

                meaning = feature_meaning.get(
                    f,
                    "No description available."
                )

                st.write(
                    f"🔹 **{f}** → {meaning}"
                )

        except Exception as e:

            st.warning("SHAP explanation failed")
            st.write(e)

        # Recommendation

        rec = recommendations.get(
            result.upper(),
            "No recommendation available."
        )

        st.info(f"💡 Recommendation: {rec}")

# -------------------------------
# IMAGE UPLOAD
# -------------------------------

if option == "Upload Image":

    st.subheader("🖼️ Upload Space Image")

    # Load class names
    if os.path.exists("assets/class_names.json"):

        with open(
            "assets/class_names.json",
            "r"
        ) as f:

            class_names = json.load(f)

    else:

        class_names = list(
            object_info.keys()
        )

    img_file = st.file_uploader(
        "Upload Image",
        type=["jpg","png","jpeg"]
    )

    if img_file:

        image = Image.open(img_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            width=300
        )

        # Check CNN model
        if cnn_model is None:

            st.error(
                "❌ CNN model not loaded. "
                "Make sure 'cnn_model.h5' exists."
            )

        else:

            try:

                # -----------------------
                # IMAGE PREPROCESS
                # -----------------------

                img = image.resize((224,224))

                img_array = np.array(img)

                img_array = img_array / 255.0

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                # -----------------------
                # PREDICT
                # -----------------------

                pred = cnn_model.predict(
                    img_array,
                    verbose=0
                )

                confidence = float(
                    np.max(pred)
                )

                class_index = int(
                    np.argmax(pred)
                )

                # Debug output
                st.write(
                    "🔍 Raw Prediction:",
                    pred
                )

                # -----------------------
                # UNKNOWN DETECTION
                # -----------------------

                UNKNOWN_THRESHOLD = 0.50

                if confidence < UNKNOWN_THRESHOLD:

                    st.error(
                        "❌ Unknown Object Detected"
                    )

                    st.warning(
                        "Model confidence too low."
                    )

                else:

                    result = class_names[
                        class_index
                    ]

                    # -----------------------
                    # RESULT DISPLAY
                    # -----------------------

                    st.success(
                        f"🌌 Prediction: "
                        f"{result.upper()}"
                    )

                    st.write(
                        f"Confidence: "
                        f"{confidence*100:.2f}%"
                    )

                    st.progress(confidence)

                    # -----------------------
                    # OBJECT INFO
                    # -----------------------

                    st.subheader(
                        "📖 What is this object?"
                    )

                    description = object_info.get(
                        result.lower(),
                        "Information not available."
                    )

                    st.info(description)

                    # -----------------------
                    # TOP 3 PREDICTIONS
                    # -----------------------

                    st.subheader(
                        "📊 Top Predictions"
                    )

                    top_indices = pred[0].argsort()[-3:][::-1]

                    top_results = []

                    for i in top_indices:

                        top_results.append({

                            "Class":
                            class_names[i],

                            "Confidence":
                            float(pred[0][i])

                        })

                    top_df = pd.DataFrame(
                        top_results
                    )

                    st.dataframe(top_df)

                    # -----------------------
                    # EXPLORE LINKS
                    # -----------------------

                    st.subheader("🔭 Explore More")

                    wiki_links = {

                        "galaxy":
                        "https://en.wikipedia.org/wiki/Galaxy",

                        "nebula":
                        "https://en.wikipedia.org/wiki/Nebula",

                        "mars":
                        "https://en.wikipedia.org/wiki/Mars",

                        "jupiter":
                        "https://en.wikipedia.org/wiki/Jupiter"

                    }

                    if result.lower() in wiki_links:

                        st.markdown(

                            f"🌐 Learn more: "
                            f"[Click here]"
                            f"({wiki_links[result.lower()]})"

                        )

            except Exception as e:

                st.error("❌ Prediction failed")

                st.write(e)# -------------------------------
# FOOTER
# -------------------------------

st.markdown("---")

st.markdown(
    "🚀 Built with AI + Streamlit"
)