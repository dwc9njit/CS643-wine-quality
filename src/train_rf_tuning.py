"""
Hyperparameter tuning for a Random Forest model in Spark.
"""

import logging
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from utils import get_spark_session, load_and_prepare_data

TRAINING_DATA_PATH = "s3a://dwc9-wine-data-1/TrainingDataset.csv"
MODEL_OUTPUT_PATH = "s3a://dwc9-wine-data-1/models/tuned_rf_model"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function for hyperparameter tuning.
    """
    # Get pre-configured SparkSession
    spark = get_spark_session("Random Forest Hyperparameter Tuning")

    try:
        logger.info("Loading and preparing dataset from S3.")
        dataset = load_and_prepare_data(spark, TRAINING_DATA_PATH)
        logger.info("Dataset schema: %s", dataset.schema.simpleString())
    except Exception as e:
        logger.error("Error loading dataset: %s", e)
        raise

    logger.info("Setting up Random Forest model and parameter grid.")
    rf = RandomForestClassifier(labelCol="quality", featuresCol="features")
    param_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [10, 50, 100])
        .addGrid(rf.maxDepth, [5, 10, 15])
        .build()
    )

    evaluator = MulticlassClassificationEvaluator(
        labelCol="quality", predictionCol="prediction", metricName="accuracy"
    )
    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=5
    )

    try:
        logger.info("Performing hyperparameter tuning with cross-validation.")
        model = cv.fit(dataset)

        # Log the best model's parameters and metrics
        best_model = model.bestModel
        best_params = best_model.extractParamMap()
        logger.info("Best model parameters: %s", best_params)
        accuracy = evaluator.evaluate(best_model.transform(dataset))
        logger.info("Best model accuracy: %.2f%%", accuracy * 100)

        # Save the best model to S3
        logger.info(f"Saving best model to {MODEL_OUTPUT_PATH}.")
        best_model.write().overwrite().save(MODEL_OUTPUT_PATH)
    except Exception as e:
        logger.error("Error during model training or saving: %s", e)
        raise

if __name__ == "__main__":
    main()
