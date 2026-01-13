import numpy as np
import random
import pandas as pd

# Import for solutions

import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, monotonically_increasing_id
from typing import Dict, Any


# function that picks a pair of random samples from the list of samples given
# (it also makes sure they do not have the same x)
def get_random_sample_pair(samples):
    dx = 0
    selected_samples = []
    while (dx == 0):
        # keep going until we get a pair with dx != 0
        selected_samples = []
        for i in [0, 1]:
            index = random.randint(0, len(samples) - 1)
            x = samples[index]['x']
            y = samples[index]['y']
            selected_samples.append({'x': x, 'y': y})
            # print("creator_samples ",i, " : ", creator_samples, " index ", index)
        dx = selected_samples[0]['x'] - selected_samples[1]['x']
    return selected_samples[0], selected_samples[1]


# generate a line model (a,b) from a pair of (x,y) samples
#  is a dictionary with x and y keys
def modelFromSamplePair(sample1, sample2):
    dx = sample1['x'] - sample2['x']
    if dx == 0:  # avoid division by zero later
        dx = 0.0001

    # model = <a,b> where y = ax+b
    # so given x1,y1 and x2,y2 =>
    #  y1 = a*x1 + b
    #  y2 = a*x2 + b
    #  y1-y2 = a*(x1 - x2) ==>  a = (y1-y2)/(x1-x2)
    #  b = y1 - a*x1

    a = (sample1['y'] - sample2['y']) / dx
    b = sample1['y'] - sample1['x'] * a
    return {'a': a, 'b': b}


# create a fit score between a list of samples and a model (a,b) - with the given cutoff distance
def scoreModelAgainstSamples(model, samples, cutoff_dist=20):
    # predict the y using the model and x samples, per sample, and sum the abs of the distances between the real y
    # with truncation of the error at distance cutoff_dist

    totalScore = 0
    for sample_i in range(0, len(samples) - 1):
        sample = samples[sample_i]
        pred_y = model['a'] * sample['x'] + model['b']
        score = min(abs(sample['y'] - pred_y), cutoff_dist)
        totalScore += score

    # print("model ",model, " score ", totalScore)
    return totalScore


# the function that runs the ransac algorithm (serially)
# gets as input the number of iterations to use and the cutoff_distance for the fit score
def ransac(samples, iterations, cutoff_dist):
    # runs ransac algorithm for the given amount of iterations, where in each iteration it:
    # 1. randomly creates a model from the samples by calling m = modelFromSamplesFunc(samples)
    # 2. calculating the score of the model against the sample set
    # 3. keeps the model with the best score
    # after all iterations are done - returns the best model and score

    min_m = {}
    min_score = -1
    for i in range(1, iterations):
        if i % 10 == 0:
            print(i)
        sample1, sample2 = get_random_sample_pair(samples)
        m = modelFromSamplePair(sample1, sample2)
        score = scoreModelAgainstSamples(m, samples, cutoff_dist)

        if (min_score < 0 or score < min_score):
            min_score = score
            min_m = m

    return {'model': min_m, 'score': min_score}


# =========== solution =================


# Ransac with pySpark implement with RDD first solution


def ransac_with_spark(samples, iterations, cutoff_dist):
    # Init pySpark context
    sc = SparkContext("local[*]", "Ransac With Spark")

    # Init Rdd with samples randomly selected from the data about the number of iterations
    paired_list = [get_random_sample_pair(samples) for _ in range(iterations)]
    rdd_of_samples = sc.parallelize(paired_list)

    # Apply the Ransac functions on the RDD
    rdd_of_models = rdd_of_samples.map(lambda x: modelFromSamplePair(*x))
    rdd_of_scores = rdd_of_models.map(lambda x: scoreModelAgainstSamples(x, samples, cutoff_dist))

    # Find the model with the minimum score without collecting
    index_rdd_scores = rdd_of_scores.zipWithIndex()
    min_score, min_idx = index_rdd_scores.reduce(lambda x, y: y if x[0] > y[0] else x)
    min_model = rdd_of_models.take(min_idx + 1)[-1]

    # Close Spark Context
    sc.stop()

    return {'model': min_model, 'score': min_score}


# Ransac with pySpark implement with DataFrame final solution


def read_csv_to_dataframe(session: SparkSession, csv_path: str) -> DataFrame:
    """
    Read a CSV file into a PySpark DataFrame.

    Parameters:
    - csv_path (str): The path to the CSV file.

    Returns:
    - DataFrame: The PySpark DataFrame containing the data from the CSV file.
    """
    df = session.read.csv(csv_path, header=True)
    df.cache()
    df = df.repartition(8)
    df.count()
    return df

def get_sampled_dataframes(df: DataFrame, iterations: int) -> (DataFrame, DataFrame):
    """
    Get two sampled DataFrames for RANSAC algorithm.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - iterations (int): The number of iterations.

    Returns:
    - DataFrame: Two sampled DataFrames (sampled_df1, sampled_df2).
    """
    count_row = df.count()
    frac = min((round(iterations / count_row, 2)) + 0.05, 1.0)

    sampled_df1 = df.sample(False, frac, seed=42).limit(iterations).select(col("x").alias("x1"), col("y").alias("y1"))
    sampled_df2 = df.sample(False, frac, seed=43).limit(iterations).select(col("x").alias("x2"), col("y").alias("y2"))

    sampled_df1 = sampled_df1.withColumn("id", monotonically_increasing_id())
    sampled_df2 = sampled_df2.withColumn("id", monotonically_increasing_id())

    return sampled_df1, sampled_df2


def perform_ransac_iterations(sampled_df1: DataFrame, sampled_df2: DataFrame, df: DataFrame,
                              cutoff_dist: float) -> DataFrame:
    """
    Perform RANSAC iterations.

    Parameters:
    - sampled_df1 (DataFrame): Sampled DataFrame 1.
    - sampled_df2 (DataFrame): Sampled DataFrame 2.
    - df (DataFrame): The original DataFrame.
    - cutoff_dist (float): The cutoff distance.

    Returns:
    - DataFrame: The resulting DataFrame after RANSAC iterations.
    """
    pairs_df = sampled_df1.join(sampled_df2, on="id", how="inner").drop("id")
    pairs_df.show()

    pairs_df = pairs_df.withColumn('dx', F.when(col("x1") - col("x2") != 0, col("x1") - col("x2")).otherwise(0.0001))
    pairs_df = pairs_df.withColumn('a', (col("y1") - col("y2")) / col("dx"))
    pairs_df = pairs_df.withColumn('b', col("y1") - (col("x1") * col("a"))).drop("x1", "y1", "x2", "y2", "dx")
    print(pairs_df.count())
    pairs_df = pairs_df.dropDuplicates(['a', 'b'])
    print(pairs_df.count())

    df = df.crossJoin(pairs_df).withColumn("pred_y", (col("a") * col("x") + col("b")))
    df = df.withColumn("score",
                       F.when(F.abs(col("y") - col("pred_y")) < cutoff_dist, F.abs(col("y") - col("pred_y"))).otherwise(
                           cutoff_dist))

    return df


def calculate_total_score(df: DataFrame) -> DataFrame:
    """
    Calculate the total score.

    Parameters:
    - df (DataFrame): The DataFrame with RANSAC results.

    Returns:
    - DataFrame: The DataFrame with total scores.
    """
    total_score_df = df.groupBy("a", "b").agg(F.sum("score").alias("total_score"))
    return total_score_df


def find_min_score_and_model(total_score_df: DataFrame) -> Dict[str, Any]:
    """
    Find the minimum score and corresponding model.

    Parameters:
    - total_score_df (DataFrame): The DataFrame with total scores.

    Returns:
    - Dict[str, Any]: Dictionary containing the minimum model and score.
    """
    min_raw = total_score_df.select(
        F.min(F.struct("total_score", *(x for x in total_score_df.columns if x != "total_score"))))
    min_raw = min_raw.first()[0]
    min_model = {"a": min_raw[1], "b": min_raw[2]}
    min_score = min_raw[0]

    return {'model': min_model, 'score': min_score}


def ransac_with_spark_df(session: SparkSession, csv_path: str, iterations: int, cutoff_dist: float) -> Dict[str, Any]:
    """
    Perform RANSAC algorithm using PySpark.

    Parameters:
    - csv_path (str): The path to the CSV file.
    - iterations (int): The number of RANSAC iterations.
    - cutoff_dist (float): The cutoff distance for scoring.

    Returns:
    - Dict[str, Any]: Dictionary containing the minimum model and score.
    """
    # Read CSV into DataFrame
    df = read_csv_to_dataframe(session, csv_path)

    # Get sampled DataFrames
    sampled_df1, sampled_df2 = get_sampled_dataframes(df, iterations)

    # Perform RANSAC iterations
    df = perform_ransac_iterations(sampled_df1, sampled_df2, df, cutoff_dist)

    # Calculate total score
    total_score_df = calculate_total_score(df)

    # Find min score and model
    result = find_min_score_and_model(total_score_df)

    return result



# ========= utility functions ============


def read_samples(filename):
    # reads samples from a csv file and returns them as list of sample dictionaries (each sample is dictionary with 'x' and 'y' keys)

    df = pd.read_csv(filename)
    samples = df[['x', 'y']].to_dict(orient='records')
    return samples


def generate_samples(n_samples=1000, n_outliers=50, b=1, output_path=None):
    # generates new samples - samples will consist of n_samples around some line + n_outliers that are not around the same line
    # gets as parameters:
    # n_samples: the number of inlier samples
    # n_outliers: the number of outlier samples
    # b: the b of the line to use ( the slope - a - will be generated randomly)
    # output_path: optional parameter for also writing out the samples into csv

    from sklearn import linear_model, datasets
    X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
                                          n_informative=1, noise=10,
                                          coef=True, bias=b)

    print(
        "generated samples around model: a = {} b = {} with {} samples + {} outliers".format(coef.item(0), b, n_samples,
                                                                                             n_outliers))
    if n_outliers > 0:
        # Add outlier data
        np.random.seed(0)
        X[:n_outliers] = 2 * np.random.normal(size=(n_outliers, 1))
        y[:n_outliers] = 10 * np.random.normal(size=n_outliers)

    d = {'x': X.flatten(), 'y': y.flatten()}
    df = pd.DataFrame(data=d)
    samples = []
    for i in range(0, len(X) - 1):
        samples.append({'x': X[i][0], 'y': y[i]})
    ref_model = {'a': coef.item(0), 'b': b}

    if not output_path is None:
        import os
        file_name = os.path.join(output_path, "samples_for_line_a_{}_b_{}.csv".format(coef.item(0), b))
        print(file_name)
        df.to_csv(file_name)
    return samples, coef, ref_model


def plot_model_and_samples(model, samples):
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [15, 8]
    plt.figure()
    xs = [s['x'] for s in samples]
    ys = [s['y'] for s in samples]
    x_min = min(xs)
    x_max = max(xs)
    y_min = model['model']['a'] * x_min + model['model']['b']
    y_max = model['model']['a'] * x_max + model['model']['b']
    plt.plot(xs, ys, '.', [x_min, x_max], [y_min, y_max], '-r')
    plt.grid()
    plt.show()


# ======== some basic pyspark example ======
def some_basic_pyspark_example():
    from pyspark import SparkContext
    num_cores_to_use = 8  # depends on how many cores you have locally. try 2X or 4X the amount of HW threads

    # now we create a spark context in local mode (i.e - not on cluster)
    sc = SparkContext("local[{}]".format(num_cores_to_use), "My First App")

    # function we will use in parallel
    def square_num(x):
        return x * x

    rdd_of_num = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

    rdd_of_num_squared = rdd_of_num.map(square_num)

    sum_of_squares = rdd_of_num_squared.reduce(lambda a, b: a + b)

    # if you want to use the DataFrame interface - you need to create a SparkSession around the spark context:
    from pyspark.sql import SparkSession
    session = SparkSession(sc)

    # create dataframe from the rdd of the numbers (call the column my_numbers)
    df = session.createDataFrame(rdd_of_num, ['my_numbers'])
    df = df.withColumn('squares', df['my_numbers'] * df['my_numbers'])
    sum_of_squares = df['squared'].sum()


# ========= main ==============

def main(session: SparkSession, path_to_samples_csv: str, num_of_iterations: int):
    import time
    start = time.time()
    cutoff_dist = 20
    best_model = ransac_with_spark_df(session=session, csv_path=path_to_samples_csv,
                                      iterations=num_of_iterations, cutoff_dist=cutoff_dist)
    print("Elapsed time: {:.2f} seconds".format(time.time() - start))
    print("Best Model:", best_model['model'])
    print("Best Score:", best_model['score'])

    # plot the model
    samples = read_samples(path_to_samples_csv)
    plot_model_and_samples(best_model, samples)


if __name__ == '__main__':
    main()
    # generate new samples
    # generate_samples(100, 5, 1, "./")
