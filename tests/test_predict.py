"""
Unit tests for predict module.
"""

import os
from pyspark.ml.classification import RandomForestClassificationModel
from utils import load_and_prepare_data

MODEL_PATH = "models/tuned_rf_model"
VALIDATION_DATA_PATH = "data/ValidationDataset.csv"

def test_prediction_loading(spark_session):
    """
    Test loading of the trained model.
    """
    # Check if model path exists
    assert os.path.exists(MODEL_PATH), f"Model path {MODEL_PATH} does not exist. Ensure the model is trained and available."

    dataset = load_and_prepare_data(spark_session, VALIDATION_DATA_PATH)
    model = RandomForestClassificationModel.load(MODEL_PATH)
    predictions = model.transform(dataset)
    assert predictions is not None, "Model failed to generate predictions."


def test_prediction_logic(spark_session):
    """
    Test prediction logic.
    """
    # Check if model path exists
    assert os.path.exists(MODEL_PATH), f"Model path {MODEL_PATH} does not exist. Ensure the model is trained and available."

    dataset = load_and_prepare_data(spark_session, VALIDATION_DATA_PATH)
    model = RandomForestClassificationModel.load(MODEL_PATH)
    predictions = model.transform(dataset)

    # Check if predictions have been made
    assert predictions is not None, "Predictions were not generated."
    # Further logic tests can be added, e.g., checking specific columns in predictions.
