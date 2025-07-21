from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression

spark = SparkSession.builder.appName("SentimentStream").getOrCreate()

# Schema for socket data
schema = StructType().add("text", StringType()).add("label", StringType())

# Read from socket
raw_df = spark.readStream.format("socket").option("host", "localhost").option("port", 9999).load()

# Parse JSON strings
from pyspark.sql.functions import from_json, col
parsed_df = raw_df.select(from_json(col("value"), schema).alias("data")).select("data.*")

# Fit ML model on static training data
train_data = spark.createDataFrame([
    ("Stock prices rose today.", "positive"),
    ("The economy is struggling.", "negative"),
    ("There is uncertainty in the market.", "neutral"),
    ("Profits are up significantly.", "positive"),
    ("Layoffs are expected.", "negative"),
], ["text", "label"])

# Label indexer
indexer = StringIndexer(inputCol="label", outputCol="label_index")

# Feature engineering
tokenizer = Tokenizer(inputCol="text", outputCol="words")
tf = HashingTF(inputCol="words", outputCol="rawFeatures")
idf = IDF(inputCol="rawFeatures", outputCol="features")

# Model
lr = LogisticRegression(featuresCol="features", labelCol="label_index")

pipeline = Pipeline(stages=[indexer, tokenizer, tf, idf, lr])
model = pipeline.fit(train_data)

# Apply model to streaming data
tokenized = tokenizer.transform(parsed_df)
tf_data = tf.transform(tokenized)
idf_model = idf.fit(tf_data)
features_df = idf_model.transform(tf_data)

predictions = model.transform(features_df)

# Show predictions in console
query = predictions.select("text", "prediction").writeStream \
    .outputMode("append").format("console").start()

query.awaitTermination()
