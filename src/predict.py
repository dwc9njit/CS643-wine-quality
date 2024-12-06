"""
Make predictions using a trained model.
"""

import logging
import os
from pyspark.ml.classification import RandomForestClassificationModel
from utils import load_and_prepare_data, get_spark_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function for making predictions.
    """
    # Get pre-configured SparkSession
    spark = get_spark_session("Make Predictions")

    logger.info("Loading and preparing dataset from S3.")
    
    # Load validation data from S3
    validation_data_path = "s3a://dwc9-wine-data-1/datasets/ValidationDataset.csv"
    dataset = load_and_prepare_data(spark, validation_data_path)

    # Load the trained model from S3
    model_path = "s3a://dwc9-wine-data-1/models/tuned_rf_model"
    model = RandomForestClassificationModel.load(model_path)

    # Make predictions
    logger.info("Making predictions.")
    predictions = model.transform(dataset)

    # Convert `features` column to string to save as CSV
    predictions = predictions.withColumn("features", predictions["features"].cast("string"))

    # Show predictions
    predictions.show()

    # Save predictions to S3
    predictions_output_path = "s3a://dwc9-wine-data-1/predictions/predictions.csv"
    logger.info(f"Saving predictions to {predictions_output_path}.")
    predictions.select("features", "quality", "prediction").write.csv(
        predictions_output_path, header=True, mode="overwrite"
    )

if __name__ == "__main__":
    main()
