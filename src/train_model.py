"""
Train a machine learning model for wine quality prediction.
"""

import logging
from pyspark.sql import SparkSession
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
    Main function to train a model.
    """
    # Configure Spark to connect to S3
    spark = (
        SparkSession.builder
        .appName("Wine Quality Prediction")
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY"))
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_KEY"))
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
        .getOrCreate()
    )

    logger.info("Loading and preparing dataset from S3.")

    s3_path = "s3a://dwc9-wine-data-1/TrainingDataset.csv"
    processed_data = load_and_prepare_data(spark, s3_path)

    logger.info("Dataset prepared. Model training logic goes here.")
    # Add model training logic here...

if __name__ == "__main__":
    main()
