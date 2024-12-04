"""
Tests for the train_model script.
"""

from utils import load_and_prepare_data

def test_train_model_logic(spark_session):
    """
    Test logic for training a model.
    """
    dataset = load_and_prepare_data(spark_session, "data/TrainingDataset.csv")
    assert dataset.count() > 0, "Dataset should not be empty."

    # Example test logic, replace with actual test case
    assert dataset.columns == ["features", "quality"], "Columns mismatch in dataset."
