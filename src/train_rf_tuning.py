"""
Hyperparameter tuning for a Random Forest model in Spark.
"""

import logging
import os
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from utils import get_spark_session, load_and_prepare_data
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    """
    Main function for hyperparameter tuning.
    """
    try:
        # Get pre-configured SparkSession
        spark = get_spark_session("Random Forest Hyperparameter Tuning")

        # Fetch and validate environment variables
        bucket_name = os.getenv("BUCKET_NAME")
        training_data_path = os.getenv("TRAINING_DATA_PATH")
        num_trees = os.getenv("NUM_TREES", "10,50,100")
        max_depth = os.getenv("MAX_DEPTH", "5,10,15")
        num_folds = int(os.getenv("NUM_FOLDS", 5))  # Default to 5 folds

        if not bucket_name or not training_data_path:
            logger.error("BUCKET_NAME or TRAINING_DATA_PATH is missing in environment variables.")
            raise ValueError("Missing BUCKET_NAME or TRAINING_DATA_PATH environment variables.")

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
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    try:
        logger.info("Setting up Random Forest model and parameter grid.")
        rf = RandomForestClassifier(labelCol="quality", featuresCol="features")

        # Build parameter grid
        param_grid = (
            ParamGridBuilder()
            .addGrid(rf.numTrees, [int(x) for x in num_trees.split(",")])
            .addGrid(rf.maxDepth, [int(x) for x in max_depth.split(",")])
            .build()
        )

        evaluator = MulticlassClassificationEvaluator(
            labelCol="quality", predictionCol="prediction", metricName="accuracy"
        )

        cv = CrossValidator(
            estimator=rf,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=num_folds
        )

        logger.info("Performing hyperparameter tuning with cross-validation.")
        model = cv.fit(dataset)

        # Log the best model's parameters
        best_model = model.bestModel
        logger.info("Best model parameters:")
        logger.info(f" - Number of Trees: {best_model.getNumTrees}")
        logger.info(f" - Max Depth: {best_model.getMaxDepth()}")

        # Evaluate the best model
        accuracy = evaluator.evaluate(best_model.transform(dataset))
        logger.info("Best model accuracy: %.2f%%", accuracy * 100)

        # Save the best model to S3
        logger.info(f"Saving best model to {model_output_path}.")
        best_model.write().overwrite().save(model_output_path)

    except Exception as e:
        logger.error(f"Error during model training or saving: {e}")
        raise

if __name__ == "__main__":
    main()
