import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
import streamlit as st
import requests
import pandas as pd
import pathlib

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, concat_ws, when, expr
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import sparknlp
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer, SentimentDLModel

NEWSAPI_KEY = st.secrets["NEWSAPI_KEY"]
MODEL_PATH = "saved_model/news_sentiment_model"


@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("NewsSentimentBootstrapBatchedAcc") \
        .master("local[*]") \
        .getOrCreate()

@st.cache_resource
def get_spark_nlp():
    return sparknlp.start()

def fetch_news(query, page_size=20):
    r = requests.get(
        "https://newsapi.org/v2/everything",
        params={"q": query, "language": "en", "pageSize": page_size, "apiKey": NEWSAPI_KEY}
    )
    r.raise_for_status()
    data = r.json().get("articles", [])
    return pd.DataFrame([{
        "title": a["title"],
        "description": a.get("description"),
        "content": a.get("content"),
        "publishedAt": a.get("publishedAt")
    } for a in data])


def build_pipeline():
    tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W")
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
    return Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])


def build_sentiment_pipeline():
    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")

    sentiment = SentimentDLModel.pretrained("sentimentdl_use_twitter", "en") \
        .setInputCols(["document", "token"]) \
        .setOutputCol("sentiment")

    return Pipeline(stages=[document_assembler, tokenizer, sentiment])


st.title("📰 News Sentiment Analysis (Batched Bootstrapping + Accuracy Check)")

query = st.text_input("Topic", "technology")
num_articles = st.slider("Number of articles", 5, 100, 20)
mode = st.radio("Mode", ["Bootstrap model (first run)", "Predict with trained model"])
run = st.button("Run")

if run:
    news_df = fetch_news(query, num_articles)
    if news_df.empty:
        st.warning("No articles found.")
    else:
        st.dataframe(news_df)

        spark = get_spark()
        sdf = spark.createDataFrame(news_df)
        sdf = sdf.withColumn("text", lower(concat_ws(" ", col("title"), col("description"), col("content"))))

        if mode == "Bootstrap model (first run)":
            # Batch sentiment labeling with Spark NLP
            st.info("Labeling data with pretrained sentiment model in parallel...")
            spark_nlp = get_spark_nlp()
            sentiment_pipeline = build_sentiment_pipeline()
            sentiment_model = sentiment_pipeline.fit(sdf)
            labeled_sdf = sentiment_model.transform(sdf)

            labeled_sdf = labeled_sdf.withColumn("label_str", expr("sentiment.result[0]"))

            
            labeled_sdf = labeled_sdf.withColumn(
                "label",
                when(col("label_str") == "positive", 1.0)
                .when(col("label_str") == "negative", 0.0)
                .otherwise(2.0)  
            )

            
            train_df, test_df = labeled_sdf.randomSplit([0.8, 0.2], seed=42)
            st.info("Training Spark ML sentiment classifier...")
            pipeline = build_pipeline()
            model = pipeline.fit(train_df)

            st.info("Evaluating model...")
            preds = model.transform(test_df)
            evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
            accuracy = evaluator.evaluate(preds)

            evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
            f1_score = evaluator_f1.evaluate(preds)

            st.write(f"**Test Accuracy:** {accuracy:.3f}")
            st.write(f"**F1 Score:** {f1_score:.3f}")

          
            pathlib.Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
            model.write().overwrite().save(MODEL_PATH)
            st.success(f"Model trained and saved at {MODEL_PATH}")

        elif mode == "Predict with trained model":
            if not pathlib.Path(MODEL_PATH).exists():
                st.error("No trained model found. Run in bootstrap mode first.")
            else:
                model = PipelineModel.load(MODEL_PATH)
                preds = model.transform(sdf)
                pdf_preds = preds.select("title", "prediction").toPandas()

                mapping = {0.0: "Negative", 1.0: "Positive", 2.0: "Neutral"}
                pdf_preds["sentiment"] = pdf_preds["prediction"].map(mapping)

                st.dataframe(pdf_preds)
                st.bar_chart(pdf_preds["sentiment"].value_counts())
