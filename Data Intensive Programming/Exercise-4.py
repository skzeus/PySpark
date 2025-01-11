# Databricks notebook source
# MAGIC %md
# MAGIC # COMP.CS.320 Data-Intensive Programming, Exercise 4
# MAGIC
# MAGIC This exercise is in two parts.
# MAGIC
# MAGIC - Tasks 1-4 contain task related to using the Spark machine learning (ML) library.
# MAGIC - Tasks 5-8 contain tasks for aggregated data frames with streaming data.
# MAGIC     - Task 5 is a reference task with static data, and the other tasks deal with streaming data frames.
# MAGIC
# MAGIC This is the **Python** version, switch to the Scala version if you want to do the tasks in Scala.
# MAGIC
# MAGIC Each task has its own cell for the code. Add your solutions to the cells. You are free to add more cells if you feel it is necessary. There are cells with example outputs following each task.
# MAGIC
# MAGIC Don't forget to submit your solutions to Moodle.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Some resources that can help with the tasks in this exercise:
# MAGIC
# MAGIC - The [tutorial notebook](https://adb-7895492183558578.18.azuredatabricks.net/?o=7895492183558578#notebook/2974598884121429) from our course
# MAGIC - Chapters 8 and 10 in [Learning Spark, 2nd Edition](https://learning.oreilly.com/library/view/learning-spark-2nd/9781492050032/)
# MAGIC     - There are additional code examples in the related [GitHub repository](https://github.com/databricks/LearningSparkV2).
# MAGIC     - The book related notebooks can be imported to Databricks by choosing `import` in your workspace and using the URL<br> `https://github.com/databricks/LearningSparkV2/blob/master/notebooks/LearningSparkv2.dbc`
# MAGIC - Apache Spark [Functions](https://spark.apache.org/docs/3.5.0/sql-ref-functions.html) for documentation on all available functions that can be used on DataFrames.<br>
# MAGIC   The full [Spark Python functions API listing](https://spark.apache.org/docs/3.5.0/api/python/reference/pyspark.sql/functions.html) for the functions package might have some additional functions listed that have not been updated in the documentation.

# COMMAND ----------

# some imports that might be required in the tasks

import time
from typing import List

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.rdd import RDD
from pyspark.sql import DataFrame, Row
from pyspark.sql import functions as F
from pyspark.sql.streaming import StreamingQuery
from pyspark.sql.types import DoubleType, StructField, StructType

# COMMAND ----------

# the initial data for the linear regression tasks
hugeSequenceOfXYData: List[Row] = [
    Row(9.44, 14.41), Row(0.89, 1.77), Row(8.65, 12.47), Row(10.43, 15.43), Row(7.39, 11.03), Row(10.06, 15.18), Row(2.07, 3.19), Row(1.24, 1.45),
    Row(3.84, 5.45), Row(10.78, 16.51), Row(10.23, 16.11), Row(9.32, 13.96), Row(7.98, 12.32), Row(0.99, 1.02), Row(6.85, 9.62), Row(8.59, 13.39),
    Row(7.35, 10.44), Row(9.85, 15.26), Row(4.59, 7.26), Row(2.43, 3.35), Row(1.58, 2.71), Row(1.59, 2.2), Row(2.1, 2.95), Row(0.62, 0.47),
    Row(5.65, 9.02), Row(5.9, 9.58), Row(8.5, 12.39), Row(8.74, 13.73), Row(1.93, 3.37), Row(10.22, 15.03), Row(10.25, 15.63), Row(1.97, 2.96),
    Row(8.03, 12.03), Row(2.05, 3.23), Row(0.69, 0.9), Row(7.58, 11.01), Row(9.99, 14.83), Row(10.53, 15.92), Row(6.12, 9.48), Row(1.34, 2.83),
    Row(3.87, 5.27), Row(4.98, 7.21), Row(4.72, 6.48), Row(8.15, 12.19), Row(2.37, 3.45), Row(10.19, 15.16), Row(10.28, 15.39), Row(8.6, 12.76),
    Row(7.46, 11.11), Row(0.25, 0.41), Row(6.41, 9.55), Row(10.49, 15.61), Row(5.18, 7.92), Row(3.74, 6.18), Row(6.27, 9.25), Row(7.51, 11.11),
    Row(4.07, 6.63), Row(5.17, 6.95), Row(9.61, 14.85), Row(4.17, 6.31), Row(4.12, 6.31), Row(9.22, 13.96), Row(5.54, 8.2), Row(0.58, 0.46),
    Row(10.13, 14.68), Row(0.53, 1.25), Row(6.87, 10.0), Row(7.17, 10.35), Row(0.09, -0.55), Row(10.8, 16.6), Row(10.31, 15.96), Row(4.74, 6.53),
    Row(1.6, 2.31), Row(5.45, 7.84), Row(0.65, 1.02), Row(2.89, 3.93), Row(6.28, 9.21), Row(8.59, 13.05), Row(6.6, 10.51), Row(8.42, 12.91)
]
dataRDD: RDD[Row] = spark.sparkContext.parallelize(hugeSequenceOfXYData)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1 - Linear regression - Training and test data
# MAGIC
# MAGIC ### Background
# MAGIC
# MAGIC In statistics, Simple linear regression is a linear regression model with a single explanatory variable.
# MAGIC That is, it concerns two-dimensional sample points with one independent variable and one dependent variable
# MAGIC (conventionally, the x and y coordinates in a Cartesian coordinate system) and finds a linear function (a non-vertical straight line)
# MAGIC that, as accurately as possible, predicts the dependent variable values as a function of the independent variable.
# MAGIC The adjective simple refers to the fact that the outcome variable is related to a single predictor. Wikipedia: [Simple linear regression](https://en.wikipedia.org/wiki/Simple_linear_regression)
# MAGIC
# MAGIC You are given an RDD of Rows, `dataRDD`, where the first element are the `x` and the second the `y` values.<br>
# MAGIC We are aiming at finding simple linear regression model for the dataset using Spark ML library. I.e. find a function `f` so that `y = f(x)` (for the 2-dimensional case `f(x)=ax+b`).
# MAGIC
# MAGIC ### Task instructions
# MAGIC
# MAGIC Transform the given `dataRDD` to a DataFrame `dataDF`, with two columns `X` (of type Double) and `label` (of type Double).
# MAGIC (`label` used here because that is the default dependent variable name in Spark ML library)
# MAGIC
# MAGIC Then split the rows in the data frame into training and testing data frames.

# COMMAND ----------

dataDF: DataFrame = spark.createDataFrame(dataRDD, schema=StructType([StructField("X", DoubleType(), False), StructField("label", DoubleType(), False)]))

# Split the data into training and testing datasets (roughly 80% for training, 20% for testing)
split = dataDF.randomSplit([0.8, 0.2], seed = 48)
trainingDF: DataFrame = split[0]
testDF: DataFrame = split[1]

print(f"Training set size: {trainingDF.count()}")
print(f"Test set size: {testDF.count()}")
print("Training set (showing only the first 6 points):")
trainingDF.show(6)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output for task 1 (the data splitting done by using seed value of 1 for data frames random splitting method):
# MAGIC
# MAGIC ```text
# MAGIC Training set size: 67
# MAGIC Test set size: 13
# MAGIC Training set (showing only the first 6 points):
# MAGIC +----+-----+
# MAGIC |   X|label|
# MAGIC +----+-----+
# MAGIC |0.89| 1.77|
# MAGIC |1.24| 1.45|
# MAGIC |2.07| 3.19|
# MAGIC |3.84| 5.45|
# MAGIC |8.65|12.47|
# MAGIC |9.44|14.41|
# MAGIC +----+-----+
# MAGIC only showing top 6 rows
# MAGIC ```
# MAGIC
# MAGIC Your output does not have to match this exactly, not even with the sizes of the training and test sets.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2 - Linear regression - Training the model
# MAGIC
# MAGIC To be able to use the ML algorithms in Spark, the input data must be given as a vector in one column. To make it easy to transform the input data into this vector format, Spark offers VectorAssembler objects.
# MAGIC
# MAGIC - Create a `VectorAssembler` for mapping the input column `X` to `features` column. And apply it to training data frame, `trainingDF,` in order to create an assembled training data frame.
# MAGIC - Then create a `LinearRegression` object. And use it with the assembled training data frame to train a linear regression model.

# COMMAND ----------

vectorAssembler: VectorAssembler = VectorAssembler(inputCols=["X"], outputCol="features")

assembledTrainingDF: DataFrame = vectorAssembler.transform(trainingDF)

assembledTrainingDF.printSchema()
assembledTrainingDF.show(6)

# COMMAND ----------

lr: LinearRegression = LinearRegression(featuresCol="features", labelCol="label")

# you can print explanations for all the parameters that can be used for linear regression by uncommenting the following:
# print(lr.explainParams())

lrModel: LinearRegressionModel = lr.fit(assembledTrainingDF)

# print out a sample of the predictions
lrModel.summary.predictions.show(6)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example outputs for task 2:
# MAGIC
# MAGIC ```text
# MAGIC root
# MAGIC  |-- X: double (nullable = false)
# MAGIC  |-- label: double (nullable = false)
# MAGIC  |-- features: vector (nullable = true)
# MAGIC
# MAGIC +----+-----+--------+
# MAGIC |   X|label|features|
# MAGIC +----+-----+--------+
# MAGIC |0.89| 1.77|  [0.89]|
# MAGIC |1.24| 1.45|  [1.24]|
# MAGIC |2.07| 3.19|  [2.07]|
# MAGIC |3.84| 5.45|  [3.84]|
# MAGIC |8.65|12.47|  [8.65]|
# MAGIC |9.44|14.41|  [9.44]|
# MAGIC +----+-----+--------+
# MAGIC only showing top 6 rows
# MAGIC ```
# MAGIC
# MAGIC and
# MAGIC
# MAGIC ```text
# MAGIC +----+-----+--------+------------------+
# MAGIC |   X|label|features|        prediction|
# MAGIC +----+-----+--------+------------------+
# MAGIC |0.89| 1.77|  [0.89]|1.2631112943485854|
# MAGIC |1.24| 1.45|  [1.24]| 1.794035541443395|
# MAGIC |2.07| 3.19|  [2.07]| 3.053084470268229|
# MAGIC |3.84| 5.45|  [3.84]|5.7380442341476945|
# MAGIC |8.65|12.47|  [8.65]| 13.03446031565065|
# MAGIC |9.44|14.41|  [9.44]|14.232832187664647|
# MAGIC +----+-----+--------+------------------+
# MAGIC only showing top 6 rows
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3 - Linear regression - Test the model
# MAGIC
# MAGIC Apply the trained linear regression model from task 2 to the test dataset.
# MAGIC
# MAGIC Then calculate the RMSE (root mean square error) for the test dataset predictions using `RegressionEvaluator` from Spark ML library.
# MAGIC
# MAGIC (The Python cell after the example output can be used to visualize the linear regression tasks)

# COMMAND ----------

testPredictions: DataFrame = lrModel.transform(vectorAssembler.transform(testDF))

testPredictions.show(6)


# all shown setters are optional, since they use the default values
testEvaluator: RegressionEvaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="rmse")

# you can print explanations for all the parameters that can be used for the regression evaluator by uncommenting the following:
# print(testEvaluator.explainParams())

testError: float = testEvaluator.evaluate(testPredictions)
print(f"The RMSE for the model is {testError}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output for task 3:
# MAGIC
# MAGIC ```text
# MAGIC +----+-----+--------+------------------+
# MAGIC |   X|label|features|        prediction|
# MAGIC +----+-----+--------+------------------+
# MAGIC |7.39|11.03|  [7.39]|11.123133026109334|
# MAGIC |7.35|10.44|  [7.35]|  11.0624559692985|
# MAGIC |1.59|  2.2|  [1.59]|2.3249597885382047|
# MAGIC | 5.9| 9.58|   [5.9]| 8.862912659905717|
# MAGIC |0.69|  0.9|  [0.69]|0.9597260102944085|
# MAGIC |9.99|14.83|  [9.99]|15.067141718813636|
# MAGIC +----+-----+--------+------------------+
# MAGIC only showing top 6 rows
# MAGIC
# MAGIC The RMSE for the model is 0.3873862940376186
# MAGIC ```
# MAGIC
# MAGIC 0 for RMSE would indicate perfect fit and the more deviations there are the larger the RMSE will be.
# MAGIC
# MAGIC You can try a different seed for dividing the data and different parameters for the linear regression to get different results.

# COMMAND ----------

# Visualization of the linear regression exercise (can be done since there are only limited number of source data points)
# Using matplotlib library to plot the data points and the prediction line
from matplotlib import pyplot

training_data = trainingDF.collect()
test_data = testDF.collect()
test_predictions = testPredictions.collect()

x_values_training = [row[0] for row in training_data]
y_values_training = [row[1] for row in training_data]
x_values_test = [row[0] for row in test_data]
y_values_test = [row[1] for row in test_data]
x_predictions = [row[0] for row in test_predictions]
y_predictions = [row[3] for row in test_predictions]

pyplot.xlabel("x")
pyplot.ylabel("y")
pyplot.xlim(min(x_values_training+x_values_test), max(x_values_training+x_values_test))
pyplot.ylim(min(y_values_training+y_values_test), max(y_values_training+y_values_test))
pyplot.plot(x_values_training, y_values_training, 'bo')
pyplot.plot(x_values_test, y_values_test, 'go')
pyplot.plot(x_predictions, y_predictions, 'r-')
pyplot.legend(["training data", "test data", "prediction line"])
pyplot.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4 - Linear regression - Making new predictions
# MAGIC
# MAGIC Use the trained `LinearRegressionModel` from task 2 to predict the `y` values for the following `x` values: `-2.8`, `3.14`, `9.9`, `123.45`

# COMMAND ----------

#Let's first start by creating a new dataframe with the provided x values:
new_x_values = [-2.8, 3.14, 9.9, 123.45]
new_data = [Row(X=x) for x in new_x_values]
xValuesDF = spark.createDataFrame(new_data)

#Transform thee DF into a vector of features
xValuesVectorized: VectorAssembler = vectorAssembler.transform(xValuesDF)

#Call the model to make predictions
predictions: DataFrame = lrModel.transform(xValuesVectorized)
predictions.show(6)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output for task 4:
# MAGIC
# MAGIC ```text
# MAGIC +------+------------------+
# MAGIC |     X|        prediction|
# MAGIC +------+------------------+
# MAGIC |  -2.8|-4.334347196450978|
# MAGIC |  3.14| 4.676195739958076|
# MAGIC |   9.9|14.930618340989255|
# MAGIC |123.45| 187.1776133627482|
# MAGIC +------+------------------+
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5 - Best selling days with static data
# MAGIC
# MAGIC This task is mostly to create a reference to which the results from the streaming tasks (6-8) can be compared to.
# MAGIC
# MAGIC In the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) the file `exercises/ex4/static/superstore_sales.csv` contains sales data from a retailer. The data is based on a dataset from [https://www.kaggle.com/datasets/soumyashanker/superstore-us](https://www.kaggle.com/datasets/soumyashanker/superstore-us).
# MAGIC
# MAGIC - Read the data from the CSV file into data frame called `salesDF`.
# MAGIC - Calculate the total sales for each day, and show the eight best selling days.
# MAGIC     - Each row has a sales record for a specific product.
# MAGIC     - The column `productPrice` contains the price for an individual product.
# MAGIC     - The column `productCount` contains the count for how many items were sold in the given sale.

# COMMAND ----------

file = "abfss://shared@tunics320f2024gen2.dfs.core.windows.net/exercises/ex4/static/superstore_sales.csv"
salesDF = spark.read  \
  .option("header", "true") \
  .option("sep", ";") \
  .option("inferSchema", "true") \
  .csv(file)
bestDaysDF: DataFrame = salesDF.withColumn("Total Sales", F.col("productPrice") * F.col("productCount"))
bestDaysDF : DataFrame = bestDaysDF.groupBy("orderDate").agg(F.sum("Total Sales").alias("dailyTotalSales")).orderBy(F.col("dailyTotalSales").desc()).limit(8)

bestDaysDF.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output for task 5:
# MAGIC
# MAGIC ```text
# MAGIC +----------+------------------+
# MAGIC | orderDate|        totalSales|
# MAGIC +----------+------------------+
# MAGIC |2014-03-18|          52148.32|
# MAGIC |2014-09-08|           22175.9|
# MAGIC |2017-11-04|          19608.77|
# MAGIC |2016-11-25|19608.739999999998|
# MAGIC |2016-10-02|18580.819999999996|
# MAGIC |2017-10-22|          18317.99|
# MAGIC |2016-05-23|16754.809999999998|
# MAGIC |2015-09-17|           16601.4|
# MAGIC +----------+------------------+
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 6 - Setting up a streaming data frame
# MAGIC
# MAGIC ### Streaming data simulation
# MAGIC
# MAGIC In this exercise, tasks 6-8, streaming data is simulated by copying CSV files from a source folder to a target folder. The target folder can thus be considered as streaming data with new file appearing after each file is copied.<br>
# MAGIC A helper function to handle the file copying is given in task 8 and you don't need to worry about that.<br>
# MAGIC The source folder from which the files are copied is in the shared container, and the target folder will be in [Students container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/students/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) where everyone has write permissions.
# MAGIC
# MAGIC ### The task
# MAGIC
# MAGIC Create a streaming data frame for similar retailer sales data as was used in task 5. The streaming data frame should point to the folder in the students container, i.e., to the address given by `myStreamingFolder`.
# MAGIC
# MAGIC Hint: Spark cannot infer the schema of streaming data, so you have to give it explicitly. You can assume that the streaming data will have the same format as the static data used in task 5.<br>
# MAGIC Also, in this case the streaming data will not contain header rows, unlike the static data used in task 5.

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, DateType
from pyspark.sql import DataFrame, Row
# identifier for your target folder to separate your streaming test from the others running at the same time
# this should only contain alphanumeric characters or underscores
myStreamingIdentifier: str = "walid"

# setup the address for your folder in the students container in the Azure storage
myStreamingFolder: str = f"abfss://students@tunics320f2024gen2.dfs.core.windows.net/ex4/walid_bayoud/"
# Ensure the folder exists in the Azure storage when creating the streaming data frame
dbutils.fs.mkdirs(myStreamingFolder)

schema = StructType([
    StructField("orderId", StringType(), True),
    StructField("orderDate", DateType(), True),
    StructField("customerId", StringType(), True),
    StructField("country", StringType(), True),
    StructField("city", StringType(), True),
    StructField("productId", StringType(), True),
    StructField("category", StringType(), True),
    StructField("subCategory", StringType(), True),
    StructField("productPrice", DoubleType(), True),
    StructField("productCount", IntegerType(), True),
    StructField("shipDate", DateType(), True),
    StructField("shipMode", StringType(), True),
])

salesStreamingDF: DataFrame = spark.readStream \
    .schema(schema) \
    .option("sep", ";") \
    .csv(myStreamingFolder)

# COMMAND ----------

# MAGIC %md
# MAGIC Note that you cannot really test this task before you have also done the tasks 7 and 8. I.e. there is no checkable output from this task.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 7 - Best selling days with streaming data
# MAGIC
# MAGIC Find the best selling days using the streaming data frame from task 6.<br>
# MAGIC This time also include the count for the total number of individual items sold on each day in the result.
# MAGIC
# MAGIC Note that in this task with the streaming data you don't need to limit the result only to the best eight selling days like was done in task 5.

# COMMAND ----------

from pyspark.sql import functions as F
bestDaysStreamingDF: DataFrame = salesStreamingDF.withColumn("Total Sales", (F.col("productPrice") * F.col("productCount")))

bestSellingDaysDF = bestDaysStreamingDF.groupBy("orderDate").agg(
    F.sum("Total Sales").alias("dailyTotalSales"),
    F.sum("productCount").alias("totalItemsSold")
)

bestSellingDaysDF = bestSellingDaysDF.orderBy(F.col("dailyTotalSales").desc())

# COMMAND ----------

# MAGIC %md
# MAGIC Note that you cannot really test this task before you have also done the task 8. I.e. there is no checkable output from this task.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 8 - Running streaming query
# MAGIC
# MAGIC Test your streaming data solution from tasks 6 and 7 by creating and starting a streaming query.
# MAGIC
# MAGIC The cell below this one contains helper scripts for copying the files (to simulate the streaming data) and to show the state of the query after each file has been copied.<br>
# MAGIC The processing time for the streaming query might depend on the number of active users in Databricks. You might need to adjust the delays defined in the `copyFiles` function to get the system working as intended.<br>
# MAGIC At least some of the slowness is caused by the given helper function system that uses separate SQL queries and the show method. See the note at the end of the example output about the reason for the used system.

# COMMAND ----------

def showQueryState(streamingQueryName: str) -> None:
    spark \
        .sql(f"SELECT * from {streamingQueryName}") \
        .show(8)

def removeFiles(folder: str) -> None:
    try:
        files = dbutils.fs.ls(folder)
        for file in files:
            dbutils.fs.rm(file.path)
    except Exception:
        # the folder did not exist => do nothing
        pass

def copyFiles(streamingQueryName: str, targetFolder: str) -> None:
    # There can be delays with the streaming data frame processing. You can try to adjust these wait times if you want.
    waitAfterFirstCopy: int = 15
    normalTimeInterval: int = 10
    postLoopWaitTime: int = 10

    streamingSource: str = "abfss://shared@tunics320f2024gen2.dfs.core.windows.net/exercises/ex4/streaming"
    inputFileList = dbutils.fs.ls(streamingSource)

    for index, csvFile in enumerate(inputFileList):
        inputFilePath: str = csvFile.path
        inputFile: str = inputFilePath.split("/")[-1]
        outputFilePath: str = targetFolder + inputFile

        # copy file from the shared container to the students container
        dbutils.fs.cp(inputFilePath, outputFilePath)
        waitTime: int = waitAfterFirstCopy if index == 0 else normalTimeInterval
        print(f"Copied file {inputFile} ({index + 1}/{len(inputFileList)}) to {outputFilePath} - waiting for {waitTime} seconds")
        time.sleep(waitTime)

        # show the current state of the streaming query
        showQueryState(streamingQueryName)

    print(f"Waiting additional {postLoopWaitTime} seconds")
    time.sleep(postLoopWaitTime)

# COMMAND ----------

# remove all files from myStreamingFolder before starting the streaming query to have a fresh run each time
removeFiles(myStreamingFolder)

# set up a unique identifier for the streaming query defined below
streamQueryName: str = f"ex4_{myStreamingIdentifier}"


# start the streaming query with the memory format and the query name set to the value of `streamQueryName`
# Define the name of the streaming query
streamQueryName = "best_selling_days_query"

# Start the streaming query
myStreamingQuery: StreamingQuery = bestSellingDaysDF.writeStream \
    .outputMode("complete") \
    .format("memory") \
    .queryName(streamQueryName) \
    .start()
# myStreamingQuery: StreamingQuery = ???


# call the helper function to copy files to the target folder to simulate a streaming data
# the helper function will also show the current state of the streaming query after each copied file
copyFiles(streamQueryName, myStreamingFolder)

# show the final state of the streaming query
print("Final state of the streaming query before stopping:")
showQueryState(streamQueryName)

# stop the streaming query to avoid the notebook running indefinitely
myStreamingQuery.stop()

# COMMAND ----------

# MAGIC %md
# MAGIC If all copied files were handled by the streaming query, the final state should match the output from task 5 (the `orderDate` and the `totalPrice` columns apart from possible rounding errors).
# MAGIC
# MAGIC The target folder is dependent on `myStreamingIdentifier` and should not match what is shown here.
# MAGIC
# MAGIC ##### Example output for task 8:
# MAGIC
# MAGIC ```text
# MAGIC Copied file 0001.csv (1/10) to abfss://students@tunics320f2024gen2.dfs.core.windows.net/ex4/streaming_test/0001.csv - waiting for 15 seconds
# MAGIC +----------+------------------+----------+
# MAGIC | orderDate|        totalPrice|totalCount|
# MAGIC +----------+------------------+----------+
# MAGIC |2014-07-26|          10931.66|        11|
# MAGIC |2017-12-07| 9454.390000000001|         8|
# MAGIC |2016-05-07|            4199.9|        10|
# MAGIC |2017-09-10|3943.4700000000003|         8|
# MAGIC |2017-09-09|3848.2000000000003|        15|
# MAGIC |2017-07-14|           3737.88|        12|
# MAGIC |2015-05-22|3716.6500000000005|         7|
# MAGIC |2016-12-03|3501.5699999999997|        12|
# MAGIC +----------+------------------+----------+
# MAGIC only showing top 8 rows
# MAGIC
# MAGIC Copied file 0002.csv (2/10) to abfss://students@tunics320f2024gen2.dfs.core.windows.net/ex4/streaming_test/0002.csv - waiting for 10 seconds
# MAGIC +----------+------------------+----------+
# MAGIC | orderDate|        totalPrice|totalCount|
# MAGIC +----------+------------------+----------+
# MAGIC |2017-11-17|          11052.78|        15|
# MAGIC |2014-07-26|11041.289999999999|        18|
# MAGIC |2017-12-07| 9454.390000000001|         8|
# MAGIC |2016-04-16| 9153.849999999999|        13|
# MAGIC |2017-08-17|           8917.08|        21|
# MAGIC |2017-10-13| 7259.849999999999|        10|
# MAGIC |2015-01-28|           7162.74|        13|
# MAGIC |2015-12-15| 6498.539999999999|        11|
# MAGIC +----------+------------------+----------+
# MAGIC only showing top 8 rows
# MAGIC
# MAGIC ...
# MAGIC ...
# MAGIC ...
# MAGIC
# MAGIC Copied file 0010.csv (10/10) to abfss://students@tunics320f2024gen2.dfs.core.windows.net/ex4/streaming_test/0010.csv - waiting for 10 seconds
# MAGIC +----------+------------------+----------+
# MAGIC | orderDate|        totalPrice|totalCount|
# MAGIC +----------+------------------+----------+
# MAGIC |2014-03-18|          52148.32|        48|
# MAGIC |2014-09-08|           22175.9|       112|
# MAGIC |2017-11-04|          19608.77|        58|
# MAGIC |2016-11-25|19608.739999999998|        53|
# MAGIC |2016-10-02|18580.819999999992|        27|
# MAGIC |2017-10-22|18317.989999999994|        49|
# MAGIC |2016-05-23|16754.809999999998|        23|
# MAGIC |2015-09-17|           16601.4|        86|
# MAGIC +----------+------------------+----------+
# MAGIC only showing top 8 rows
# MAGIC
# MAGIC Waiting additional 10 seconds
# MAGIC Final state of the streaming query before stopping:
# MAGIC +----------+------------------+----------+
# MAGIC | orderDate|        totalPrice|totalCount|
# MAGIC +----------+------------------+----------+
# MAGIC |2014-03-18|          52148.32|        48|
# MAGIC |2014-09-08|           22175.9|       112|
# MAGIC |2017-11-04|          19608.77|        58|
# MAGIC |2016-11-25|19608.739999999998|        53|
# MAGIC |2016-10-02|18580.819999999992|        27|
# MAGIC |2017-10-22|18317.989999999994|        49|
# MAGIC |2016-05-23|16754.809999999998|        23|
# MAGIC |2015-09-17|           16601.4|        86|
# MAGIC +----------+------------------+----------+
# MAGIC only showing top 8 rows
# MAGIC ```
# MAGIC
# MAGIC Note that the `awaitTermination` function of the streaming query could be used to show the console outputs in Databricks, but it's not used in this exercise to avoid notebooks being left running for an indefinitely long time.<br>
# MAGIC Similarly, the Databricks `display` command is not used here due to similar reasons.
# MAGIC
# MAGIC The Scala version of the exercise uses awaitTermination with timeout parameter and console output, but the exact same functionality could not be easily implemented with Python.
# MAGIC
# MAGIC You are free to test the other ways to display streaming queries, but just be sure to stop all cells in the notebook once you are done. Closing the browser tab or the entire browser will not stop a running streaming query!