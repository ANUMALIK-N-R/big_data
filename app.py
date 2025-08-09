import streamlit as st
import pandas as pd
import requests
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import lit

@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("NewsSentimentLight") \
        .master("local[*]") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

spark = get_spark()


NEWS_API_KEY = st.secrets["news_api_key"]

def fetch_news(topic, page_size=5):
    url = f"https://newsapi.org/v2/everything?q={topic}&pageSize={page_size}&apiKey={NEWS_API_KEY}"
    r = requests.get(url)
    articles = r.json().get("articles", [])
    return pd.DataFrame([{"title": a["title"]} for a in articles if a.get("title")])

st.title("📰 Minimal PySpark News Sentiment")

topic = st.text_input("Topic", "technology")
mode = st.radio("Mode", ["Bootstrap", "Predict"])
num_articles = st.slider("Number of articles", 3, 20, 5)

if st.button("Run"):
    df_pd = fetch_news(topic, num_articles)
    if df_pd.empty:
        st.error("No news found.")
    else:
        df_spark = spark.createDataFrame(df_pd)

        if mode == "Bootstrap":
            # Fake sentiment labels for demo
            df_spark = df_spark.withColumn("label", lit(1.0))  # all positive just to train
            
            tokenizer = Tokenizer(inputCol="title", outputCol="words")
            words_data = tokenizer.transform(df_spark)

            hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
            featurized_data = hashingTF.transform(words_data)

            idf = IDF(inputCol="rawFeatures", outputCol="features")
            idf_model = idf.fit(featurized_data)
            rescaled_data = idf_model.transform(featurized_data)

            lr = LogisticRegression(maxIter=10, regParam=0.001)
            model = lr.fit(rescaled_data)

            st.session_state["model"] = model
            st.session_state["hashingTF"] = hashingTF
            st.session_state["tokenizer"] = tokenizer
            st.session_state["idf_model"] = idf_model

            st.success("Model trained in Spark (labels are fake for demo).")

        elif mode == "Predict":
            if "model" not in st.session_state:
                st.error("Run Bootstrap first!")
            else:
                tokenizer = st.session_state["tokenizer"]
                hashingTF = st.session_state["hashingTF"]
                idf_model = st.session_state["idf_model"]
                model = st.session_state["model"]

                words_data = tokenizer.transform(df_spark)
                featurized_data = hashingTF.transform(words_data)
                rescaled_data = idf_model.transform(featurized_data)
                predictions = model.transform(rescaled_data)

                st.write(predictions.select("title", "prediction").toPandas())
