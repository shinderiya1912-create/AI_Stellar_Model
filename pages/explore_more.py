import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import shap
import matplotlib.pyplot as plt
import joblib
import os
import xml.etree.ElementTree as ET
from datetime import datetime   # ⭐ FIXED

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Explore More",
    layout="wide"
)

st.title("🔭 Explore More – Stellar Data")

# -------------------------------
# LOAD MODEL + SCALER
# -------------------------------
def load_file(file):
    if os.path.exists(file):
        return joblib.load(file)
    else:
        st.error(f"❌ Missing file: {file}")
        st.stop()

model = load_file("stellar_rf_model.pkl")
scaler = load_file("scaler.pkl")

# -------------------------------
# SIDEBAR MENU
# -------------------------------
st.sidebar.title("🔭 Explore Menu")

explore_option = st.sidebar.radio(
    "🔭 Choose Feature",
    [
        "📊 Feature Exploration",
        "📈 Correlation Analysis",
        "⭐ Feature Importance Viewer",
        "🌍 NASA APOD",
        "🛰️ NASA Live News",
    ]
)

# -------------------------------
# FEATURE LIST
# -------------------------------
feature_names = [
    'alpha','delta','u','g','r','i','z','redshift',
    'run_ID','rerun_ID','cam_col','field_ID',
    'spec_obj_ID','plate','fiber_ID','MJD'
]

# -------------------------------
# LOAD DATA
# -------------------------------
data = pd.read_csv(
    r"C:\MSc.FY Projects\Stellar Object Prediction\sdss_data\star_classification1.csv"
)

# ==================================================
# FEATURE EXPLORATION
# ==================================================
if explore_option == "📊 Feature Exploration":

    st.header("📊 Feature Exploration")

    selected_feature = st.selectbox(
        "Select Feature to Explore",
        feature_names
    )

    fig = px.histogram(
        data,
        x=selected_feature,
        title=f"Distribution of {selected_feature}"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# CORRELATION ANALYSIS
# ==================================================
elif explore_option == "📈 Correlation Analysis":

    st.header("📈 Correlation Analysis")

    import seaborn as sns

    numeric_data = data.select_dtypes(include=['number'])

    if numeric_data.empty:

        st.error("No numeric columns found.")

    else:

        corr = numeric_data.corr()

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)

# ==================================================
# FEATURE IMPORTANCE
# ==================================================
elif explore_option == "⭐ Feature Importance Viewer":

    st.header("⭐ Feature Importance Viewer")

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    top_n = st.slider(
        "Select Number of Top Features",
        5,
        len(feature_names),
        10
    )

    importance_df = importance_df.head(top_n)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance Ranking"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Importance Table")

    st.dataframe(
        importance_df,
        use_container_width=True
    )

    top_feature = importance_df.iloc[0]["Feature"]

    st.success(
        f"🔥 Most Important Feature: {top_feature}"
    )

# ==================================================
# NASA APOD (FIXED + DATE SAVE)
# ==================================================
elif explore_option == "🌍 NASA APOD":

    st.header("🌍 NASA Astronomy Picture of the Day")

    # ⭐ DATE INPUT FIXED
    selected_date = st.date_input(
        "📅 Select Date",
        datetime.today()
    )

    url = "https://api.nasa.gov/planetary/apod"

    params = {
        "api_key": "DEMO_KEY",
        "date": str(selected_date)
    }

    r = requests.get(url, params=params)

    if r.status_code == 200:

        data = r.json()

        st.subheader(data["title"])

        if data["media_type"] == "image":

            st.image(
                data["url"],
                use_container_width=True
            )

        elif data["media_type"] == "video":

            st.video(data["url"])

        st.write(data["explanation"])

        # ⭐ SAVE DATE + TIME
        now = datetime.now()

        record = {
            "Selected_Date": str(selected_date),
            "Fetched_Time": now.strftime("%H:%M:%S"),
            "Title": data.get("title", ""),
            "Image_URL": data.get("url", "")
        }

        file_name = "nasa_apod_log.csv"

        df_new = pd.DataFrame([record])

        if os.path.exists(file_name):

            df_old = pd.read_csv(file_name)

            df_all = pd.concat(
                [df_old, df_new],
                ignore_index=True
            )

        else:

            df_all = df_new

        df_all.to_csv(
            file_name,
            index=False
        )

        st.success("✅ APOD Saved to Log File")

    else:

        st.error("❌ Failed to load APOD.")

# ==================================================
# NASA LIVE NEWS
# ==================================================
elif explore_option == "🛰️ NASA Live News":

    st.header("🛰️ NASA Live News")

    rss_url = "https://www.nasa.gov/rss/dyn/breaking_news.rss"

    response = requests.get(rss_url)

    if response.status_code == 200:

        root = ET.fromstring(response.content)

        items = root.findall(".//item")

        st.success("Latest NASA Updates")

        for item in items[:8]:

            title = item.find("title").text
            link = item.find("link").text

            st.markdown(
                f'<a href="{link}" target="_blank">🔗 {title}</a>',
                unsafe_allow_html=True
            )

    else:

        st.error("❌ Unable to load NASA News.")