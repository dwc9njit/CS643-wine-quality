"""
Train a Random Forest model for wine quality prediction.
"""

import logging
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from utils import load_and_prepare_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to train a Random Forest model.
    """
    # Configure Spark to connect to S3
    spark = (
        SparkSession.builder
        .appName("Wine Quality Prediction")
        .config("spark.hadoop.fs.s3a.access.key", "REDACTED")
        .config("spark.hadoop.fs.s3a.secret.key", "REDACTED")
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
        .getOrCreate()
    )
    
    logger.info("Loading and preparing dataset from S3.")

    # Load training data from S3
    training_data_path = "s3a://dwc9-wine-data-1/TrainingDataset.csv"
    dataset = load_and_prepare_data(spark, training_data_path)

    # Train Random Forest model
    logger.info("Training Random Forest model.")
    rf = RandomForestClassifier(labelCol="quality", featuresCol="features", numTrees=10)
    model = rf.fit(dataset)
    logger.info("Random Forest model trained successfully.")

    # Save the trained model to S3
    model_output_path = "s3a://dwc9-wine-data-1/models/tuned_rf_model"
    logger.info(f"Saving model to {model_output_path}.")
    model.write().overwrite().save(model_output_path)

    # Model evaluation
    logger.info("Evaluating the model.")
    predictions = model.transform(dataset)
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    logger.info("Model accuracy: %.2f%%", accuracy * 100)

if __name__ == "__main__":
    main()
