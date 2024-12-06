"""
Train a Random Forest model for wine quality prediction.
"""

import logging
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from utils import get_spark_session, load_and_prepare_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to train a Random Forest model.
    """
    # Get pre-configured SparkSession
    spark = get_spark_session("Train Random Forest Model")

    logger.info("Loading and preparing dataset from S3.")
    
    # S3 path for training data
    training_data_path = "s3a://dwc9-wine-data-1/TrainingDataset.csv"
    
    # Load and prepare data
    dataset = load_and_prepare_data(spark, training_data_path)

    logger.info("Training Random Forest model.")
    rf = RandomForestClassifier(labelCol="quality", featuresCol="features", numTrees=10)
    model = rf.fit(dataset)
    logger.info("Random Forest model trained successfully.")

    # Save the trained model to S3
    model_output_path = "s3a://dwc9-wine-data-1/models/tuned_rf_model"
    logger.info(f"Saving model to {model_output_path}.")
    model.write().overwrite().save(model_output_path)

    # Evaluate the model
    logger.info("Evaluating the model.")
    predictions = model.transform(dataset)
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    logger.info("Model accuracy: %.2f%%", accuracy * 100)

if __name__ == "__main__":
    main()
