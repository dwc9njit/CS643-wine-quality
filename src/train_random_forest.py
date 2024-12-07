"""
Train a Random Forest model for wine quality prediction.
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
    Main function to train a Random Forest model.
    """
    try:
        # Get pre-configured SparkSession
        spark = get_spark_session("Train Random Forest Model")

        logger.info("Loading and preparing dataset from S3.")

        # Fetch and validate environment variables
        bucket_name = os.getenv("BUCKET_NAME")
        training_data_path = os.getenv("TRAINING_DATA_PATH")
        num_trees = int(os.getenv("NUM_TREES", 10))  # Default to 10 trees if not specified

        if not bucket_name or not training_data_path:
            logger.error("BUCKET_NAME or TRAINING_DATA_PATH is missing in environment variables.")
            raise ValueError("Missing BUCKET_NAME or TRAINING_DATA_PATH environment variables.")

        # Construct S3 path for training data
        training_data_s3_path = f"s3a://{bucket_name}/{training_data_path}"

        # Load and prepare data
        dataset = load_and_prepare_data(spark, training_data_s3_path)

        # Validate dataset
        if dataset.rdd.isEmpty():
            logger.error("The dataset is empty. Please check the data source.")
            raise ValueError("Dataset is empty.")

        logger.info("Training Random Forest model.")
        rf = RandomForestClassifier(
            labelCol="quality",
            featuresCol="features",
            numTrees=num_trees
        )
        model = rf.fit(dataset)
        logger.info("Random Forest model trained successfully.")

        # Save the trained model to S3
        model_output_path = f"s3a://{bucket_name}/models/tuned_rf_model"
        logger.info(f"Saving model to {model_output_path}.")
        model.write().overwrite().save(model_output_path)

        # Evaluate the model
        logger.info("Evaluating the model.")
        predictions = model.transform(dataset)
        evaluator = MulticlassClassificationEvaluator(
            labelCol="quality",
            predictionCol="prediction",
            metricName="accuracy"
        )
        accuracy = evaluator.evaluate(predictions)
        logger.info(f"Model accuracy: {accuracy * 100:.2f}%")

        # Save evaluation metrics locally
        metrics_file = "model_accuracy.txt"
        with open(metrics_file, "w") as f:
            f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n")

        # Upload metrics file to S3
        metrics_output_path = f"s3a://{bucket_name}/metrics/model_accuracy.txt"
        logger.info(f"Uploading model accuracy to {metrics_output_path}.")
        spark.sparkContext.binaryFiles(metrics_file).saveAsTextFile(metrics_output_path)

        logger.info("Model training and evaluation completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
        raise

if __name__ == "__main__":
    main()
