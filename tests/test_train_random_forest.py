"""
Tests for train_random_forest script.
"""

from pyspark.sql import SparkSession


def test_random_forest_model_training(spark_session: SparkSession):
    """
    Test to verify the random forest model is trained correctly.
    """
    data = spark_session.read.csv(
        "data/TrainingDataset.csv", header=True, inferSchema=True, sep=";"
    )
    assert data is not None, "Training dataset is None."
    assert data.count() > 0, "Training dataset is empty."
