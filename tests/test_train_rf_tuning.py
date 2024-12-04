"""
Unit tests for the train_rf_tuning module.
"""

import pytest
from pyspark.ml.classification import RandomForestClassificationModel

def test_tuning_model_loading(spark_session):
    """
    Test to ensure the tuned random forest model loads correctly.
    """
    try:
        model_path = "models/tuned_rf_model"
        model = RandomForestClassificationModel.load(model_path)
        assert model is not None, f"Failed to load model from {model_path}"
    except FileNotFoundError as e:
        pytest.fail(f"Model file not found: {e}")
    except ValueError as e:
        pytest.fail(f"Value error while loading model: {e}")
