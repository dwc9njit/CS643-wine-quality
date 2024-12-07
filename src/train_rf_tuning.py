"""
Hyperparameter tuning for a Random Forest model in Spark.
"""

import os
import logging
from dotenv import load_dotenv
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from utils import get_spark_session, load_and_prepare_data

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_env_variables():
    """
    Validates required environment variables and returns their values.
    """
    bucket_name = os.getenv("BUCKET_NAME")
    training_data_path = os.getenv("TRAINING_DATA_PATH")
    num_trees = os.getenv("NUM_TREES", "10,50,100").split(",")
    max_depth = os.getenv("MAX_DEPTH", "5,10,15").split(",")
    num_folds = int(os.getenv("NUM_FOLDS", "5"))

    if not bucket_name or not training_data_path:
        logger.error("BUCKET_NAME or TRAINING_DATA_PATH is missing in environment variables.")
        raise ValueError("Missing BUCKET_NAME or TRAINING_DATA_PATH environment variables.")

    return bucket_name, training_data_path, num_trees, max_depth, num_folds


def build_param_grid(rf_model, num_trees, max_depth):
    """
    Builds the parameter grid for hyperparameter tuning.
    """
    return (
        ParamGridBuilder()
        .addGrid(rf_model.numTrees, [int(x) for x in num_trees])
        .addGrid(rf_model.maxDepth, [int(x) for x in max_depth])
        .build()
    )


def main():
    """
    Main function for hyperparameter tuning.
    """
    try:
        # Get pre-configured SparkSession
        spark = get_spark_session("Random Forest Hyperparameter Tuning")

        # Fetch environment variables
        bucket_name, training_data_path, num_trees, max_depth, num_folds = validate_env_variables()

        # Construct S3 paths
        training_data_s3_path = f"s3a://{bucket_name}/{training_data_path}"
        model_output_path = f"s3a://{bucket_name}/models/tuned_rf_model"

        logger.info("Loading and preparing dataset from S3.")
        dataset = load_and_prepare_data(spark, training_data_s3_path)

        # Validate dataset
        if dataset.rdd.isEmpty():
            logger.error("The dataset is empty. Please check the data source.")
            raise ValueError("Dataset is empty.")

        logger.info("Dataset schema: %s", dataset.schema.simpleString())

        # Setting up Random Forest model and parameter grid
        rf = RandomForestClassifier(labelCol="quality", featuresCol="features")
        param_grid = build_param_grid(rf, num_trees, max_depth)

        evaluator = MulticlassClassificationEvaluator(
            labelCol="quality", predictionCol="prediction", metricName="accuracy"
        )

        cv = CrossValidator(
            estimator=rf,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=num_folds,
        )

        logger.info("Performing hyperparameter tuning with cross-validation.")
        model = cv.fit(dataset)

        # Log the best model's parameters
        best_model = model.bestModel
        logger.info("Best model parameters:")
        logger.info(" - Number of Trees: %s", best_model.getNumTrees)
        logger.info(" - Max Depth: %s", best_model.getMaxDepth())

        # Evaluate the best model
        accuracy = evaluator.evaluate(best_model.transform(dataset))
        logger.info("Best model accuracy: %.2f%%", accuracy * 100)

        # Save the best model to S3
        logger.info("Saving best model to %s", model_output_path)
        best_model.write().overwrite().save(model_output_path)

    except Exception as e:
        logger.error("Error during hyperparameter tuning or saving: %s", str(e))
        raise


if __name__ == "__main__":
    main()
