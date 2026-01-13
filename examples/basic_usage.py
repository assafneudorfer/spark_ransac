#!/usr/bin/env python3
"""
Basic usage example for Distributed RANSAC with PySpark

This script demonstrates how to:
1. Initialize a Spark session
2. Run distributed RANSAC on sample data
3. Visualize the results
"""

import sys
import os
import time

# Add parent directory to path to import spark_ransac
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import SparkSession
from spark_ransac import ransac_with_spark_df, read_samples, plot_model_and_samples


def main():
    """Run a simple RANSAC example"""

    # Initialize Spark Session
    print("Initializing Spark Session...")
    spark = SparkSession.builder \
        .appName("RANSAC Example") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # Set log level to reduce noise
    spark.sparkContext.setLogLevel("WARN")

    # Configuration
    data_path = './data/samples_for_line_a_27.0976088174_b_12.234.csv'
    iterations = 5000
    cutoff_distance = 20

    print(f"\nRunning Distributed RANSAC:")
    print(f"  Dataset: {data_path}")
    print(f"  Iterations: {iterations}")
    print(f"  Cutoff Distance: {cutoff_distance}")
    print()

    # Run distributed RANSAC
    start_time = time.time()
    result = ransac_with_spark_df(
        session=spark,
        csv_path=data_path,
        iterations=iterations,
        cutoff_dist=cutoff_distance
    )
    elapsed_time = time.time() - start_time

    # Display results
    print(f"\nResults:")
    print(f"  Best Model: y = {result['model']['a']:.4f}x + {result['model']['b']:.4f}")
    print(f"  Score: {result['score']:.2f}")
    print(f"  Execution Time: {elapsed_time:.2f} seconds")

    # Visualize (optional)
    try:
        print("\nGenerating visualization...")
        samples = read_samples(data_path)
        plot_model_and_samples(result, samples)
    except Exception as e:
        print(f"Visualization skipped: {e}")

    # Cleanup
    spark.stop()
    print("\nDone!")


if __name__ == '__main__':
    main()
