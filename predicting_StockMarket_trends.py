# LAB EXPERIMENT: Predicting Stock Market Trends using Big Data (PySpark)
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Step 1: Initialize Spark Session
spark = SparkSession.builder \
 .appName("StockMarketTrendPrediction") \
 .getOrCreate()

# Step 2: Create synthetic big stock market dataset
data = spark.createDataFrame([
 (100, 105, 98, 104, 200000),
 (104, 108, 102, 107, 220000),
 (107, 109, 105, 106, 250000),
 (106, 110, 104, 109, 300000),
 (109, 112, 108, 111, 280000),
 (111, 113, 109, 110, 260000)
] * 1000, # simulate big data
["open", "high", "low", "close", "volume"])
print("Total records:", data.count())

# Step 3: Feature Engineering – Daily Return & Trend
data = data.withColumn(
 "return",
 (col("close") - col("open")) / col("open")
)
data = data.withColumn(
 "trend",
 when(col("return") > 0, 1).otherwise(0)
)

# Step 4: Feature Vector Assembly
assembler = VectorAssembler(
 inputCols=["open", "high", "low", "close", "volume", "return"],
 outputCol="features"
)
data_features = assembler.transform(data)

# Step 5: Feature Scaling
scaler = StandardScaler(
 inputCol="features",
 outputCol="scaled_features",
 withMean=True,
 withStd=True
)
scaler_model = scaler.fit(data_features)
data_scaled = scaler_model.transform(data_features)

# Step 6: Train-Test Split
train_data, test_data = data_scaled.randomSplit([0.7, 0.3], seed=42)

# Step 7: LogisƟc Regression Model
lr = LogisticRegression(
 featuresCol="scaled_features",
 labelCol="trend"
)
model = lr.fit(train_data)

# Step 8: Predictions
predictions = model.transform(test_data)

# Step 9: Model Evaluation
evaluator = BinaryClassificationEvaluator(
 labelCol="trend",
 rawPredictionCol="rawPrediction",
 metricName="areaUnderROC"
)
auc = evaluator.evaluate(predictions)
print("Area Under ROC:", auc)

# Stop Spark session
spark.stop()
