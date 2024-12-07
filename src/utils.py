from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_spark_session(app_name):
    """
    Create and return a SparkSession configured for S3 with DirectWrite.
    Handles missing AWS credentials gracefully for debugging in local environments.
    """
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_KEY")

    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        logger.error("AWS credentials are missing. Check AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
        raise ValueError("Missing AWS credentials.")

    try:
        spark = (
            SparkSession.builder
            .appName(app_name)
            .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY)
            .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_KEY)
            .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1,com.amazonaws:aws-java-sdk-bundle:1.12.530")
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
            .config("spark.hadoop.fs.s3a.prefetch.enable", "false")
            .config("spark.hadoop.fs.s3a.experimental.input.fadvise", "sequential")
            .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
            .config("spark.hadoop.fs.s3a.committer.name", "directory")
            .config("spark.hadoop.fs.s3a.committer.staging.conflict-mode", "replace")
            .config("spark.hadoop.fs.s3a.committer.staging.tmp.path", "/tmp/s3a")
            .config("spark.hadoop.fs.s3a.committer.magic.enabled", "false")
            .getOrCreate()
        )

        logger.info("SparkSession created successfully.")
        return spark
    except Exception as e:
        logger.error(f"Failed to create SparkSession: {e}")
        raise

def load_and_prepare_data(spark, filepath):
    """
    Loads and prepares a dataset from a given filepath.
    Handles schema validation and feature assembly.
    """
    try:
        logger.info(f"Loading dataset from {filepath}")
        data = spark.read.csv(filepath, header=True, inferSchema=True, sep=";")
        if data.rdd.isEmpty():
            logger.error(f"Dataset at {filepath} is empty or could not be loaded.")
            raise ValueError("Dataset is empty.")

        logger.info("Dataset schema:")
        data.printSchema()

        # Validate required columns
        required_columns = {"fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                            "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
                            "pH", "sulphates", "alcohol", "quality"}
        if not required_columns.issubset(set(data.columns)):
            missing = required_columns - set(data.columns)
            logger.error(f"Missing columns in dataset: {missing}")
            raise ValueError(f"Dataset is missing required columns: {missing}")

        # Assemble features
        logger.info("Assembling features.")
        feature_columns = [col for col in data.columns if col != "quality"]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        prepared_data = assembler.transform(data).select("features", "quality")

        logger.info("Data prepared successfully.")
        return prepared_data
    except Exception as e:
        logger.error(f"Failed to load and prepare data: {e}")
        raise
