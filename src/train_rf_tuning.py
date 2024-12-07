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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function for hyperparameter tuning.
    """
    # Get pre-configured SparkSession
    spark = get_spark_session("Random Forest Hyperparameter Tuning")

    # Construct S3 paths
    training_data_path = f"s3a://{os.getenv('BUCKET_NAME')}/{os.getenv('TRAINING_DATA_PATH')}"
    model_output_path = f"s3a://{os.getenv('BUCKET_NAME')}/models/tuned_rf_model"

    try:
        logger.info("Loading and preparing dataset from S3.")
        dataset = load_and_prepare_data(spark, training_data_path)

        # Validate dataset
        if dataset.rdd.isEmpty():
            logger.error("The dataset is empty. Please check the data source.")
            return

        logger.info("Dataset schema: %s", dataset.schema.simpleString())
    except Exception as e:
        logger.error("Error loading dataset: %s", e)
        raise

    logger.info("Setting up Random Forest model and parameter grid.")
    rf = RandomForestClassifier(labelCol="quality", featuresCol="features")

    # Build parameter grid
    param_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [int(x) for x in os.getenv("NUM_TREES", "10,50,100").split(",")])
        .addGrid(rf.maxDepth, [int(x) for x in os.getenv("MAX_DEPTH", "5,10,15").split(",")])
        .build()
    )

    evaluator = MulticlassClassificationEvaluator(
        labelCol="quality", predictionCol="prediction", metricName="accuracy"
    )

    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=int(os.getenv("NUM_FOLDS", 5))  # Default to 5 folds
    )

    try:
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
        logger.error("Error during model training or saving: %s", e)
        raise

if __name__ == "__main__":
    main()
