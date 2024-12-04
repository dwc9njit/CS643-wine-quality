"""
Unit tests for predict module.
"""

from pyspark.ml.classification import RandomForestClassificationModel
from utils import load_and_prepare_data

def test_prediction_loading(spark_session):
    """
    Test loading of the trained model.
    """
    dataset = load_and_prepare_data(spark_session, "data/ValidationDataset.csv")
    model = RandomForestClassificationModel.load("models/tuned_rf_model")
    predictions = model.transform(dataset)
    assert predictions is not None


def test_prediction_logic(spark_session):
    """
    Test prediction logic.
    """
    dataset = load_and_prepare_data(spark_session, "data/ValidationDataset.csv")
    model = RandomForestClassificationModel.load("models/tuned_rf_model")
    predictions = model.transform(dataset)
    assert predictions is not None
