from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassificationModel

# Step 1: Initialize Spark session
spark = SparkSession.builder.appName("Wine Quality Prediction Validation").getOrCreate()

# Step 2: Load the validation dataset
validation_data = spark.read.csv(
    "data/ValidationDataset.csv",
    header=True,
    inferSchema=True,
    sep=";"
)

# Step 3: Preprocess the data
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
    outputCol="features")
validation_dataset = assembler.transform(validation_data).select("features", "quality")

# Step 4: Load the tuned Random Forest model
rf_model = RandomForestClassificationModel.load("models/tuned_rf_model")

# Step 5: Make predictions
predictions = rf_model.transform(validation_dataset)

# Step 6: Evaluate model performance
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="weightedPrecision")
precision = evaluator.evaluate(predictions)

evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="weightedRecall")
recall = evaluator.evaluate(predictions)

evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)

# Display the performance metrics
print(f"Model Performance Metrics (Validation Dataset):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")

# Step 7: Save the predictions
predictions.select("quality", "prediction").write.csv("data/validation_predictions_rf.csv", header=True, mode="overwrite")

# Stop the Spark session
spark.stop()
