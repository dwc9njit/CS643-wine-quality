from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Step 1: Initialize Spark session
spark = SparkSession.builder.appName("Wine Quality Prediction").getOrCreate()

# Step 2: Load the dataset
data = spark.read.csv(
    "data/TrainingDataset.csv",
    header=True,
    inferSchema=True,
    sep=";",           # Correct delimiter
    quote='"',         # Handle double quotes
    escape='"',        # Escape additional quotes
    ignoreLeadingWhiteSpace=True,
    ignoreTrailingWhiteSpace=True
)

# Clean column names by stripping excessive quotes
cleaned_columns = [col_name.strip('"') for col_name in data.columns]
data = data.toDF(*cleaned_columns)

# Verify the cleaned column names
print("Cleaned Columns:", data.columns)

# Show a sample of the dataset to verify
data.show(5)

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
    outputCol="features")


# Assemble features and select features and label (quality)
dataset = assembler.transform(data).select("features", "quality")

# Step 4: Train logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="quality")
model = lr.fit(dataset)

# Step 5: Save the trained model
model.write().overwrite().save("models/logistic_regression_model")

# Step 6: Stop the Spark session
spark.stop()
