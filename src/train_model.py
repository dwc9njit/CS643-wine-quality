"""
Train a machine learning model for wine quality prediction..
"""

import logging
import os
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from utils import get_spark_session, load_and_prepare_data
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    """
    Main function to train a Random Forest model for wine quality prediction.
    """
    try:
        # Get pre-configured SparkSession
        spark = get_spark_session("Train Machine Learning Model")

        logger.info("Loading and preparing dataset from S3.")

        # Fetch and validate environment variables
        bucket_name = os.getenv("BUCKET_NAME")
        training_data_path = os.getenv("TRAINING_DATA_PATH")
        if not bucket_name or not training_data_path:
            logger.error("BUCKET_NAME or TRAINING_DATA_PATH is missing in environment variables.")
            raise ValueError("Missing BUCKET_NAME or TRAINING_DATA_PATH environment variables.")

        # Construct S3 path for training data
        training_data_s3_path = f"s3a://{bucket_name}/{training_data_path}"

        # Load and prepare data
        processed_data = load_and_prepare_data(spark, training_data_s3_path)

        logger.info("Dataset prepared. Training the model.")

        # Train Random Forest model
        rf = RandomForestClassifier(labelCol="quality", featuresCol="features", numTrees=50, maxDepth=10)
        model = rf.fit(processed_data)

        # Evaluate the model
        logger.info("Evaluating the model.")
        predictions = model.transform(processed_data)
        evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        logger.info(f"Model accuracy: {accuracy * 100:.2f}%")

        # Save the trained model to S3
        model_output_path = f"s3a://{bucket_name}/models/tuned_rf_model"
        logger.info(f"Saving model to {model_output_path}.")
        model.write().overwrite().save(model_output_path)

        logger.info("Model training and saving completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise

if __name__ == "__main__":
    main()
