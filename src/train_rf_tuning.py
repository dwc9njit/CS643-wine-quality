"""
Hyperparameter tuning for a Random Forest model in Spark.
"""

import logging
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from utils import load_and_prepare_data
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "s3.amazonaws.com")
TRAINING_DATA_PATH = "s3a://dwc9-wine-data-1/TrainingDataset.csv"
MODEL_OUTPUT_PATH = "s3a://dwc9-wine-data-1/models/tuned_rf_model"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function for hyperparameter tuning.
    """
    # Validate AWS credentials
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        logger.error("AWS_ACCESS_KEY or AWS_SECRET_KEY is not set.")
        raise ValueError("Missing AWS credentials. Please set them in the environment.")
    
    logger.info("AWS_ACCESS_KEY: %s", AWS_ACCESS_KEY[:4] + "..." + AWS_ACCESS_KEY[-4:])

    # Configure Spark to connect to S3
    spark = (
        SparkSession.builder
        .appName("Random Forest Tuning")
        .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.endpoint", S3_ENDPOINT)
        .getOrCreate()
    )

    try:
        logger.info("Loading and preparing dataset from S3.")
        dataset = load_and_prepare_data(spark, TRAINING_DATA_PATH)
        logger.info("Dataset schema: %s", dataset.schema.simpleString())
    except Exception as e:
        logger.error("Error loading dataset: %s", e)
        raise

    logger.info("Setting up Random Forest model and parameter grid.")
    rf = RandomForestClassifier(labelCol="quality", featuresCol="features")
    param_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [10, 50, 100])
        .addGrid(rf.maxDepth, [5, 10, 15])
        .build()
    )

    evaluator = MulticlassClassificationEvaluator(
        labelCol="quality", predictionCol="prediction", metricName="accuracy"
    )
    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=5
    )

    try:
        logger.info("Performing hyperparameter tuning with cross-validation.")
        model = cv.fit(dataset)

        # Log the best model's parameters and metrics
        best_model = model.bestModel
        best_params = best_model.extractParamMap()
        logger.info("Best model parameters: %s", best_params)
        accuracy = evaluator.evaluate(best_model.transform(dataset))
        logger.info("Best model accuracy: %.2f%%", accuracy * 100)

        # Save the best model to S3
        logger.info(f"Saving best model to {MODEL_OUTPUT_PATH}.")
        best_model.write().overwrite().save(MODEL_OUTPUT_PATH)
    except Exception as e:
        logger.error("Error during model training or saving: %s", e)
        raise

if __name__ == "__main__":
    main()
