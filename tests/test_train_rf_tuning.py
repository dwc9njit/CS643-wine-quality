import pytest
from pyspark.ml.classification import RandomForestClassificationModel

def test_tuning_model_loading(spark_session):
    """
    Test to ensure the tuned random forest model loads correctly.
    """
    model_path = "models/tuned_rf_model"
    
    # Ensure the Spark context is active
    spark_session.sparkContext.setLogLevel("WARN")
    
    try:
        model = RandomForestClassificationModel.load(model_path)
        assert model is not None, f"Failed to load model from {model_path}"
    except Exception as e:
        pytest.fail(f"Model loading failed with error: {str(e)}")
