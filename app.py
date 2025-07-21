import streamlit as st
import pandas as pd
import time
import os
import glob

st.set_page_config(page_title="Live Sentiment Dashboard", layout="wide")
st.title("📈 Real-Time Sentiment Analysis Dashboard")

# Folder where Spark is writing predictions
log_dir = "predictions_log"

# Label map (update if your label index changes)
label_map = {0.0: "negative", 1.0: "neutral", 2.0: "positive"}

# Sidebar: Refresh rate
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 10, 2)

placeholder = st.empty()

while True:
    try:
        all_files = glob.glob(f"{log_dir}/*.csv")
        df_list = [pd.read_csv(f) for f in all_files if "part" in f]
        if not df_list:
            st.warning("Waiting for data...")
            time.sleep(refresh_interval)
            continue

        df = pd.concat(df_list, ignore_index=True)

        # Clean malformed rows (Spark sometimes repeats headers in files)
        df = df[df["text"] != "text"]
        df.drop_duplicates(inplace=True)

        df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
        df = df.dropna(subset=["prediction"])

        df["sentiment"] = df["prediction"].map(label_map)

        with placeholder.container():
            st.subheader("🔍 Latest Predictions")
            st.dataframe(df.tail(10), use_container_width=True)

            st.subheader("📊 Sentiment Distribution")
            sentiment_counts = df["sentiment"].value_counts()
            st.bar_chart(sentiment_counts)

        time.sleep(refresh_interval)

    except Exception as e:
        st.error(f"Error reading prediction files: {e}")
        time.sleep(refresh_interval)
