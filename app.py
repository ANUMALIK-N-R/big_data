import streamlit as st
import pandas as pd
import time
import os

st.set_page_config(page_title="Live Sentiment Dashboard", layout="wide")

st.title("📈 Real-Time Sentiment Analysis Dashboard")

log_file = "predictions_log.csv"  # Path where PySpark job writes predictions

# Initialize empty dataframe
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write("text,prediction\n")

# Label mapping
label_map = {0.0: "negative", 1.0: "neutral", 2.0: "positive"}

# Sidebar refresh
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 10, 2)

# Run loop
placeholder = st.empty()

while True:
    try:
        df = pd.read_csv(log_file)

        # Drop duplicates if any
        df = df.drop_duplicates()

        # Convert numeric prediction to label
        df["sentiment"] = df["prediction"].map(label_map)

        with placeholder.container():
            st.subheader("🔍 Latest Sentiment Predictions")
            st.dataframe(df.tail(10), use_container_width=True)

            sentiment_counts = df["sentiment"].value_counts().to_dict()

            st.subheader("📊 Sentiment Distribution")
            st.bar_chart(pd.DataFrame.from_dict(sentiment_counts, orient="index"))

        time.sleep(refresh_interval)

    except Exception as e:
        st.error(f"Error reading file: {e}")
        time.sleep(refresh_interval)
