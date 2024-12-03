from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Step 1: Initialize Spark session
spark = SparkSession.builder.appName("Wine Quality Prediction - Random Forest").getOrCreate()

# Step 2: Load the training dataset
data = spark.read.csv(
    "data/TrainingDataset.csv",
    header=True,
    inferSchema=True,
    sep=";",           # Correct delimiter
    ignoreLeadingWhiteSpace=True,
    ignoreTrailingWhiteSpace=True
)

# Step 3: Assemble features
assembler = VectorAssembler(inputCols=[
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol"
    ],
    outputCol="features"
)
dataset = assembler.transform(data).select("features", "quality")

# Step 4: Train Random Forest model
rf = RandomForestClassifier(featuresCol="features", labelCol="quality", numTrees=50, maxDepth=5, seed=42)
rf_model = rf.fit(dataset)

# Step 5: Save the trained model
rf_model.write().overwrite().save("models/random_forest_model")

# Step 6: Evaluate the model on training data
predictions = rf_model.transform(dataset)

# Evaluate using accuracy, precision, recall, and F1-score
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")

accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
f1 = evaluator.setMetricName("f1").evaluate(predictions)

# Print performance metrics
print(f"Model Performance Metrics (Training Dataset):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Step 7: Stop the Spark session
spark.stop()
