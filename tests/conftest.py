"""
Shared pytest fixtures for tests.
"""

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="module")
def spark_session():
    """
    Fixture to provide a SparkSession for testing.
    The SparkSession is created once per test session.
    """
    spark = SparkSession.builder.master("local[*]").appName("TestWineQuality").getOrCreate()
    yield spark
    spark.stop()
