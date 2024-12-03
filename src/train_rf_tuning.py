from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Step 1: Initialize Spark session
spark = SparkSession.builder.appName("Wine Quality Prediction - RF Tuning").getOrCreate()

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

# Step 4: Define the Random Forest model
rf = RandomForestClassifier(featuresCol="features", labelCol="quality", seed=42)

# Step 5: Set up the parameter grid for tuning
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 50, 100]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .addGrid(rf.minInstancesPerNode, [1, 5, 10]) \
    .build()

# Step 6: Define cross-validation
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")

crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)  # 3-fold cross-validation

# Step 7: Run cross-validation and get the best model
cvModel = crossval.fit(dataset)
bestModel = cvModel.bestModel

# Step 8: Save the best model
bestModel.write().overwrite().save("models/tuned_rf_model")

# Step 9: Evaluate the best model on the training dataset
predictions = bestModel.transform(dataset)
accuracy = evaluator.evaluate(predictions)
precision = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions)
recall = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions)
f1 = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1").evaluate(predictions)

# Print the best parameters and performance metrics
print(f"Best Model Parameters:")
print(f"  numTrees: {bestModel.getNumTrees}")
print(f"  maxDepth: {bestModel.getMaxDepth()}")
print(f"  minInstancesPerNode: {bestModel.getMinInstancesPerNode}")
print("\nModel Performance Metrics (Training Dataset):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Step 10: Stop the Spark session
spark.stop()
