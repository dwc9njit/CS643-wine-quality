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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to train a Random Forest model.
    """
    # Get pre-configured SparkSession
    spark = get_spark_session("Train Random Forest Model")

    logger.info("Loading and preparing dataset from S3.")
    
    # Construct S3 path for training data
    training_data_path = f"s3a://{os.getenv('BUCKET_NAME')}/{os.getenv('TRAINING_DATA_PATH')}"

    # Load and prepare data
    dataset = load_and_prepare_data(spark, training_data_path)

    # Validate dataset
    if dataset.rdd.isEmpty():
        logger.error("The dataset is empty. Please check the data source.")
        return

    logger.info("Training Random Forest model.")
    rf = RandomForestClassifier(
        labelCol="quality", 
        featuresCol="features", 
        numTrees=int(os.getenv("NUM_TREES", 10))  # Default to 10 if not specified
    )
    model = rf.fit(dataset)
    logger.info("Random Forest model trained successfully.")

    # Save the trained model to S3
    model_output_path = f"s3a://{os.getenv('BUCKET_NAME')}/models/tuned_rf_model"
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
    logger.info("Model accuracy: %.2f%%", accuracy * 100)

    # Save evaluation metrics
    metrics_output_path = f"s3a://{os.getenv('BUCKET_NAME')}/metrics/model_accuracy.txt"
    logger.info(f"Saving model accuracy to {metrics_output_path}.")
    with open("accuracy.txt", "w") as f:
        f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n")
    
    # Upload metrics to S3
    spark.sparkContext.addFile("accuracy.txt")
    spark.sparkContext.binaryFiles("accuracy.txt").saveAsTextFile(metrics_output_path)

if __name__ == "__main__":
    main()
