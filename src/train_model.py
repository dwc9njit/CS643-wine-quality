"""
Train a machine learning model for wine quality prediction.
"""

import logging
import os
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from utils import get_spark_session, load_and_prepare_data
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to train a model.
    """
    # Get pre-configured SparkSession
    spark = get_spark_session("Train Machine Learning Model")

    logger.info("Loading and preparing dataset from S3.")
    
    # Construct S3 path for training data
    training_data_path = f"s3a://{os.getenv('BUCKET_NAME')}/{os.getenv('TRAINING_DATA_PATH')}"

    # Load and prepare data
    processed_data = load_and_prepare_data(spark, training_data_path)

    logger.info("Dataset prepared. Training the model.")

    # Train Random Forest model
    rf = RandomForestClassifier(labelCol="quality", featuresCol="features", numTrees=50, maxDepth=10)
    model = rf.fit(processed_data)

    # Evaluate model
    logger.info("Evaluating the model.")
    predictions = model.transform(processed_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    logger.info(f"Model accuracy: {accuracy * 100:.2f}%")

    # Save the trained model to S3
    model_output_path = f"s3a://{os.getenv('BUCKET_NAME')}/models/tuned_rf_model"
    logger.info(f"Saving model to {model_output_path}.")
    model.write().overwrite().save(model_output_path)

    logger.info("Model training and saving completed successfully.")

if __name__ == "__main__":
    main()
