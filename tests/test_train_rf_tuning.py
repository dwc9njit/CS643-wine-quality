"""
Unit tests for the train_rf_tuning module.
"""

import pytest
from pyspark.ml.classification import RandomForestClassificationModel

def test_tuning_model_loading(spark_session):
    """
    Test to ensure the tuned random forest model loads correctly.
    """
    spark_session.sparkContext.setLogLevel("WARN")  # Use the spark_session here
    model_path = "models/tuned_rf_model"
    model = RandomForestClassificationModel.load(model_path)
    assert model is not None, f"Failed to load model from {model_path}"

