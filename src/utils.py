"""
Utility functions for common operations in the project.
"""

import logging
from pyspark.ml.feature import VectorAssembler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_data(spark, filepath):
    """
    Loads and prepares a dataset from a given filepath.
    """
    logger.info("Loading dataset from %s", filepath)
    data = spark.read.csv(filepath, header=True, inferSchema=True, sep=";")
    data = data.toDF(*[col.strip().replace(" ", "_") for col in data.columns])

    logger.info("Assembling features.")
    assembler = VectorAssembler(
        inputCols=[
            "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
            "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
            "pH", "sulphates", "alcohol",
        ],
        outputCol="features",
    )
    return assembler.transform(data).select("features", "quality")
