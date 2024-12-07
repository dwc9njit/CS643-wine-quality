"""
Make predictions using a trained model and validate the results.
"""

import logging
import os
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from utils import load_and_prepare_data, get_spark_session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    """
    Main function for making predictions and validating results.
    """
    try:
        # Get pre-configured SparkSession
        spark = get_spark_session("Make Predictions and Validate Model")

        # Fetch and validate environment variables
        bucket_name = os.getenv("BUCKET_NAME")
        validation_data_path = os.getenv("VALIDATION_DATA_PATH")
        if not bucket_name or not validation_data_path:
            logger.error("BUCKET_NAME or VALIDATION_DATA_PATH is missing in environment variables.")
            raise ValueError("Missing required environment variables.")

        model_path = f"s3a://{bucket_name}/models/tuned_rf_model"
        predictions_output_path = f"s3a://{bucket_name}/predictions/predictions.csv"

        # Load and prepare dataset
        logger.info("Loading and preparing dataset from S3.")
        dataset = load_and_prepare_data(spark, f"s3a://{bucket_name}/{validation_data_path}")

        if dataset.rdd.isEmpty():
            logger.error("The dataset is empty. Please check the data source.")
            raise ValueError("Validation dataset is empty.")

        logger.info("Dataset schema: %s", dataset.schema.simpleString())

    except Exception as e:
        logger.error(f"Error during dataset loading: {e}")
        return

    try:
        # Load trained model
        logger.info(f"Loading trained model from {model_path}.")
        model = RandomForestClassificationModel.load(model_path)
    except Exception as e:
        logger.error(f"Error loading trained model: {e}")
        return

    try:
        # Make predictions
        logger.info("Making predictions.")
        predictions = model.transform(dataset)

        # Include essential columns
        predictions = predictions.select(
            col("features").cast("string").alias("features"),
            "quality",
            "prediction"
        )

        # Display a sample of predictions
        logger.info("Sample predictions:")
        predictions.show(10)

        # Validate predictions
        logger.info("Validating predictions.")
        evaluator = MulticlassClassificationEvaluator(
            labelCol="quality",
            predictionCol="prediction",
            metricName="accuracy"
        )
        accuracy = evaluator.evaluate(predictions)
        logger.info(f"Model Accuracy: {accuracy:.2%}")

        # Save predictions to S3
        logger.info(f"Saving predictions to {predictions_output_path}.")
        predictions.write.csv(predictions_output_path, header=True, mode="overwrite")

        # Final summary
        total_predictions = predictions.count()
        logger.info(f"Total predictions made: {total_predictions}")
        logger.info(f"Predictions saved to: {predictions_output_path}")

    except Exception as e:
        logger.error(f"Error during prediction, validation, or saving: {e}")
        return

if __name__ == "__main__":
    main()
