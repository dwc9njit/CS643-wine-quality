"""
Make predictions using a trained model.
"""

import logging
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
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
    Main function for making predictions.
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
    
    # Load validation data from S3
    validation_data_path = "s3a://dwc9-wine-data-1/ValidationDataset.csv"
    dataset = load_and_prepare_data(spark, validation_data_path)

    # Load the trained model from S3
    model_path = "s3a://dwc9-wine-data-1/models/tuned_rf_model"
    model = RandomForestClassificationModel.load(model_path)

    # Make predictions
    logger.info("Making predictions.")
    predictions = model.transform(dataset)
    predictions.show()

    # Save predictions to S3
    predictions_output_path = "s3a://dwc9-wine-data-1/predictions/predictions.csv"
    logger.info(f"Saving predictions to {predictions_output_path}.")
    predictions.select("features", "quality", "prediction").write.csv(
        predictions_output_path, header=True, mode="overwrite"
    )

if __name__ == "__main__":
    main()
