"""
Make predictions using a trained model.
"""

import logging
import os
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql.functions import col
from utils import load_and_prepare_data, get_spark_session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function for making predictions.
    """
    # Get pre-configured SparkSession
    spark = get_spark_session("Make Predictions")

    # Construct S3 paths
    validation_data_path = f"s3a://{os.getenv('BUCKET_NAME')}/{os.getenv('VALIDATION_DATA_PATH')}"
    model_path = f"s3a://{os.getenv('BUCKET_NAME')}/models/tuned_rf_model"
    predictions_output_path = f"s3a://{os.getenv('BUCKET_NAME')}/predictions/predictions.csv"

    try:
        logger.info("Loading and preparing dataset from S3.")
        dataset = load_and_prepare_data(spark, validation_data_path)

        # Validate dataset
        if dataset.rdd.isEmpty():
            logger.error("The dataset is empty. Please check the data source.")
            return

        logger.info("Dataset schema: %s", dataset.schema.simpleString())
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    try:
        logger.info(f"Loading trained model from {model_path}.")
        model = RandomForestClassificationModel.load(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    try:
        logger.info("Making predictions.")
        predictions = model.transform(dataset)

        # Include derived columns (e.g., probabilities) if applicable
        predictions = predictions.select(
            col("features").cast("string").alias("features"),
            "quality",
            "prediction"
        )

        # Show predictions
        predictions.show()

        # Save predictions to S3
        logger.info(f"Saving predictions to {predictions_output_path}.")
        predictions.write.csv(predictions_output_path, header=True, mode="overwrite")
    except Exception as e:
        logger.error(f"Error during prediction or saving: {e}")
        return

if __name__ == "__main__":
    main()
