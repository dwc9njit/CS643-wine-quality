"""
Make predictions using a trained model.
"""

import logging
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from utils import load_and_prepare_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function for making predictions.
    """
    spark = SparkSession.builder.appName("Wine Quality Prediction").getOrCreate()
    logger.info("Loading and preparing dataset.")

    dataset = load_and_prepare_data(spark, "data/ValidationDataset.csv")
    model = RandomForestClassificationModel.load("models/tuned_rf_model")

    predictions = model.transform(dataset)
    predictions.show()

    # Save predictions to a CSV file
    logger.info("Saving predictions to 'data/predictions.csv'.")
    predictions.select("features", "quality", "prediction").write.csv(
        "data/predictions.csv", header=True, mode="overwrite")   

if __name__ == "__main__":
    main()
