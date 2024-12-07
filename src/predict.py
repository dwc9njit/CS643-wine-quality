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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function for making predictions and validating results.
    """
    # Get pre-configured SparkSession
    spark = get_spark_session("Make Predictions and Validate Model")

    # Construct S3 paths
    validation_data_path = f"s3a://{os.getenv('BUCKET_NAME')}/{os.getenv('VALIDATION_DATA_PATH')}"
    model_path = f"s3a://{os.getenv('BUCKET_NAME')}/models/tuned_rf_model"
    predictions_output_path = f"s3a://{os.getenv('BUCKET_NAME')}/predictions/predictions.csv"

    try:
        # Load and prepare dataset
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
        # Load trained model
        logger.info(f"Loading trained model from {model_path}.")
        model = RandomForestClassificationModel.load(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    try:
        # Make predictions
        logger.info("Making predictions.")
        predictions = model.transform(dataset)

        # Include derived columns (e.g., probabilities) if applicable
        predictions = predictions.select(
            col("features").cast("string").alias("features"),
            "quality",
            "prediction"
        )

        # Show a sample of predictions
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
