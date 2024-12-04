"""
Train a machine learning model for wine quality prediction.
"""

import logging
from pyspark.sql import SparkSession
from utils import load_and_prepare_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to train a model.
    """
    spark = SparkSession.builder.appName("Wine Quality Prediction").getOrCreate()
    logger.info("Loading and preparing dataset.")

    load_and_prepare_data(spark, "data/TrainingDataset.csv")
    logger.info("Dataset prepared. Model training logic goes here.")

if __name__ == "__main__":
    main()
