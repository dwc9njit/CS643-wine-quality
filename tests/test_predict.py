"""
Unit tests for the predict module.
"""

import os
from pyspark.ml.classification import RandomForestClassificationModel
from utils import load_and_prepare_data

MODEL_PATH = "models/tuned_rf_model"
VALIDATION_DATA_PATH = "data/ValidationDataset.csv"
MODEL_NOT_FOUND_ERROR = (
    f"Model path {MODEL_PATH} does not exist. Ensure the model is trained and available."
)
PREDICTIONS_NOT_GENERATED_ERROR = "Predictions were not generated."
DATASET_EMPTY_ERROR = "The validation dataset is empty. Check the input data file."
PREDICTIONS_COLUMNS_ERROR = "Predictions DataFrame is missing the 'prediction' column."


def test_prediction_loading(spark_session):
    """
    Test loading of the trained model.
    """
    # Check if model path exists
    assert os.path.exists(MODEL_PATH), MODEL_NOT_FOUND_ERROR

    dataset = load_and_prepare_data(spark_session, VALIDATION_DATA_PATH)
    assert dataset.count() > 0, DATASET_EMPTY_ERROR

    model = RandomForestClassificationModel.load(MODEL_PATH)
    predictions = model.transform(dataset)
    assert predictions is not None, PREDICTIONS_NOT_GENERATED_ERROR


def test_prediction_logic(spark_session):
    """
    Test prediction logic.
    """
    # Check if model path exists
    assert os.path.exists(MODEL_PATH), MODEL_NOT_FOUND_ERROR

    dataset = load_and_prepare_data(spark_session, VALIDATION_DATA_PATH)
    assert dataset.count() > 0, DATASET_EMPTY_ERROR

    model = RandomForestClassificationModel.load(MODEL_PATH)
    predictions = model.transform(dataset)

    # Check if predictions have been made
    assert predictions is not None, PREDICTIONS_NOT_GENERATED_ERROR

    # Verify predictions contain the required column
    assert "prediction" in predictions.columns, PREDICTIONS_COLUMNS_ERROR
    assert predictions.select("prediction").count() > 0, "No predictions were made."
