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
    spark = SparkSession.builder.appName("Wine Quality Prediction").getOrCreate()
    logger.info("Loading and preparing dataset.")

    dataset = load_and_prepare_data(spark, "data/TrainingDataset.csv")

    # Train Random Forest model
    rf = RandomForestClassifier(labelCol="quality", featuresCol="features", numTrees=10)
    model = rf.fit(dataset)
    logger.info("Random Forest model trained successfully.")

    # Model evaluation
    predictions = model.transform(dataset)
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    logger.info("Model accuracy: %.2f%%", accuracy * 100)

if __name__ == "__main__":
    main()
