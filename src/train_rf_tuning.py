"""
Hyperparameter tuning for a Random Forest model.
"""

import logging
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from utils import load_and_prepare_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function for hyperparameter tuning.
    """
    spark = SparkSession.builder.appName("Random Forest Tuning").getOrCreate()
    logger.info("Loading and preparing dataset.")

    dataset = load_and_prepare_data(spark, "data/TrainingDataset.csv")

    rf = RandomForestClassifier(labelCol="quality", featuresCol="features")
    param_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [10, 50, 100])
        .addGrid(rf.maxDepth, [5, 10, 15])
        .build()
    )

    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")
    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=5
    )


    model = cv.fit(dataset)
    best_params = model.bestModel.extractParamMap()
    logger.info("Best model parameters: %s", best_params)


if __name__ == "__main__":
    main()
