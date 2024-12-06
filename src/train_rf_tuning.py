"""
Hyperparameter tuning for a Random Forest model.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function for hyperparameter tuning.
    """
    # Configure Spark to connect to S3
    spark = (
        SparkSession.builder
        .appName("Random Forest Tuning")
        .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
        .getOrCreate()
    )

    logger.info("Loading and preparing dataset from S3.")
    
    training_data_path = "s3a://dwc9-wine-data-1/TrainingDataset.csv"
    dataset = load_and_prepare_data(spark, training_data_path)

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

    logger.info("Performing hyperparameter tuning with cross-validation.")
    model = cv.fit(dataset)

    # Log the best model's parameters
    best_model = model.bestModel
    best_params = best_model.extractParamMap()
    logger.info("Best model parameters: %s", best_params)

    # Save the best model to S3
    model_output_path = "s3a://dwc9-wine-data-1/models/tuned_rf_model"
    logger.info(f"Saving best model to {model_output_path}.")
    best_model.write().overwrite().save(model_output_path)

if __name__ == "__main__":
    main()
