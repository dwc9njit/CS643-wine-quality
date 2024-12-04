"""
Unit tests for the predict module.
"""

import os
from pyspark.ml.classification import RandomForestClassificationModel
from utils import load_and_prepare_data

MODEL_PATH = "models/tuned_rf_model"
VALIDATION_DATA_PATH = "data/ValidationDataset.csv"

ERROR_MESSAGES = {
    "model_not_found": f"Model path {MODEL_PATH} does not exist. Ensure the model is trained and available.",
    "dataset_empty": "The validation dataset is empty. Check the input data file.",
    "predictions_not_generated": "Predictions were not generated.",
    "missing_prediction_column": "Predictions DataFrame is missing the 'prediction' column.",
    "no_predictions_made": "No predictions were made.",
}

def validate_model_path():
    """
    Ensure the model path exists.
    """
    assert os.path.exists(MODEL_PATH), ERROR_MESSAGES["model_not_found"]

def validate_dataset(dataset):
    """
    Ensure the dataset is not empty.
    """
    assert dataset.count() > 0, ERROR_MESSAGES["dataset_empty"]

def validate_predictions(predictions):
    """
    Ensure predictions are generated and contain the 'prediction' column.
    """
    assert predictions is not None, ERROR_MESSAGES["predictions_not_generated"]
    assert "prediction" in predictions.columns, ERROR_MESSAGES["missing_prediction_column"]
    assert predictions.select("prediction").count() > 0, ERROR_MESSAGES["no_predictions_made"]

def test_prediction_loading(spark_session):
    """
    Test loading of the trained model and basic prediction functionality.
    """
    validate_model_path()
    dataset = load_and_prepare_data(spark_session, VALIDATION_DATA_PATH)
    validate_dataset(dataset)
    model = RandomForestClassificationModel.load(MODEL_PATH)
    predictions = model.transform(dataset)
    validate_predictions(predictions)

def test_prediction_logic(spark_session):
    """
    Test logic of predictions, ensuring valid predictions are made.
    """
    validate_model_path()
    dataset = load_and_prepare_data(spark_session, VALIDATION_DATA_PATH)
    validate_dataset(dataset)
    model = RandomForestClassificationModel.load(MODEL_PATH)
    predictions = model.transform(dataset)
    validate_predictions(predictions)

    # Additional logic checks for predictions can be added here
