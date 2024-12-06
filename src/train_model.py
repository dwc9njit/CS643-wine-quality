"""
Train a machine learning model for wine quality prediction.
"""

import logging
from pyspark.sql import SparkSession
from utils import load_and_prepare_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to train a model.
    """
    # Configure Spark to work with S3
    spark = (
        SparkSession.builder.appName("Wine Quality Prediction")
        .config("spark.hadoop.fs.s3a.access.key", "REDACTED")
        .config("spark.hadoop.fs.s3a.secret.key", "REDACTED")
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
        .getOrCreate()
    )

    logger.info("Loading and preparing dataset from S3.")

    # Replace with the path to your S3 bucket
    s3_path = "s3a://dwc9-wine-data-1/TrainingDataset.csv"
    load_and_prepare_data(spark, s3_path)

    logger.info("Dataset prepared. Model training logic goes here.")

if __name__ == "__main__":
    main()
