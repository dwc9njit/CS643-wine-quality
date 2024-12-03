from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel

# Step 1: Initialize Spark session
spark = SparkSession.builder.appName("Wine Quality Prediction Validation").getOrCreate()

# Step 2: Load the validation dataset
data = spark.read.csv(
    "data/ValidationDataset.csv",
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

# Step 4: Load the trained model
model = LogisticRegressionModel.load("models/logistic_regression_model")

# Step 5: Make predictions
predictions = model.transform(dataset)

# Step 6: Save the predictions (excluding the features column)
predictions.select("quality", "prediction").write.csv("data/predictions.csv", header=True, mode="overwrite")


# Step 7: Stop the Spark session
spark.stop()
