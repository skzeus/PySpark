# Databricks notebook source
# MAGIC %md
# MAGIC # COMP.CS.320 Data-Intensive Programming, Exercise 3
# MAGIC
# MAGIC This exercise is in three parts.
# MAGIC
# MAGIC - Tasks 1-3 contain additional tasks of data processing using Spark and DataFrames.
# MAGIC - Tasks 4-6 are basic tasks of using RDDs with textual data.
# MAGIC - Tasks 7-8 are similar basic tasks for textual data but using DataFrames instead.
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
# MAGIC - Chapters 3 and 6 in [Learning Spark, 2nd Edition](https://learning.oreilly.com/library/view/learning-spark-2nd/9781492050032/)
# MAGIC     - There are additional code examples in the related [GitHub repository](https://github.com/databricks/LearningSparkV2).
# MAGIC     - The book related notebooks can be imported to Databricks by choosing `import` in your workspace and using the URL<br> `https://github.com/databricks/LearningSparkV2/blob/master/notebooks/LearningSparkv2.dbc`
# MAGIC - Apache Spark [Functions](https://spark.apache.org/docs/3.5.0/sql-ref-functions.html) for documentation on all available functions that can be used on DataFrames.<br>
# MAGIC   The full [Spark Python functions API listing](https://spark.apache.org/docs/3.5.0/api/python/reference/pyspark.sql/functions.html) for the functions package might have some additional functions listed that have not been updated in the documentation.
# MAGIC - Apache Spark [RDD Programming Guide](https://spark.apache.org/docs/3.5.0/rdd-programming-guide.html)
# MAGIC - Apache Spark [Datasets and DataFrames](https://spark.apache.org/docs/3.5.0/sql-programming-guide.html#datasets-and-dataframes) documentation

# COMMAND ----------

# some imports that might be required in the tasks

import dataclasses
import re
from typing import List, Tuple

from pyspark.rdd import RDD
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql import Row
from pyspark.sql.types import IntegerType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1 - Load ProCem data
# MAGIC
# MAGIC ##### Background
# MAGIC
# MAGIC As part of Tampere University research projects [ProCem](https://www.senecc.fi/projects/procem-2) and [ProCemPlus](https://www.senecc.fi/projects/procemplus) various data from the Kampusareena building at Hervanta campus was gathered. In addition, data from several other sources were gathered in the projects. The other data sources included, for example, the electricity market prices and the weather measurements from a weather station located at the Sähkötalo building at Hervanta. The data gathering system developed in the projects is still running and gathering data.
# MAGIC
# MAGIC A later, still ongoing, research project [DELI](https://research.tuni.fi/tase/projects/) has as part of its agenda to research the best ways to manage and share the collected data. In the project some of the ProCem data was uploaded into a [Apache IoTDB](https://iotdb.apache.org/) instance to test how well it could be used with the data. IoTDB is a data management system for time series data. Some of the data uploaded to IoTDB is used in tasks 1-3.
# MAGIC
# MAGIC The IoTDB has a Spark connector plugin that can be used to load data from IoTDB directly into Spark DataFrame. However, to not make things too complicated for the exercise, a ready-made sample of the data has already been extracted and given as a static data for this and the following two tasks.
# MAGIC
# MAGIC ##### The data
# MAGIC
# MAGIC In the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) folder `exercises/ex3/procem_iotdb.parquet` contains some ProCem data fetched from IoTDB in Parquet format.
# MAGIC
# MAGIC Brief explanations on the columns:
# MAGIC
# MAGIC - `Time`: the UNIX timestamp in millisecond precision
# MAGIC - `SolarPower`: the total electricity power produced by the solar panels on Kampusareena (`W`)
# MAGIC - `WaterCooling01Power` and `WaterCooling02Power`: the total electricity power used by the two water cooling machineries on Kampusareena (`W`)
# MAGIC - `VentilationPower`: the total electricity power used by the ventilation machinery on Kampusareena (`W`)
# MAGIC - `Temperature`: the temperature measured by the weather station on top of Sähkötalo (`°C`)
# MAGIC - `WindSpeed`: the wind speed measured by the weather station on top of Sähkötalo (`m/s`)
# MAGIC - `Humidity`: the humidity measured by the weather station on top of Sähkötalo (`%`)
# MAGIC - `ElectricityPrice`: the market price for electricity in Finland (`€/MWh`)
# MAGIC
# MAGIC Parquet is column-oriented data file format designed for efficient data storage and retrieval. Unlike CSV files the data given in Parquet format is not as easy to preview without Spark. But if really want, for example, the [Parquet Visualizer](https://marketplace.visualstudio.com/items?itemName=lucien-martijn.parquet-visualizer) Visual Studio Code extension can be used to browse the data contained in a Parquet file. However, understanding the format is not important for this exercise.
# MAGIC
# MAGIC ##### The task
# MAGIC
# MAGIC - Read the data into a DataFrame.
# MAGIC - Print out the schema for the resulting DataFrame.
# MAGIC - Show/display at least the first 10 rows.

# COMMAND ----------

procemDF = spark.read.parquet("abfss://shared@tunics320f2024gen2.dfs.core.windows.net/exercises/ex3/procem_iotdb.parquet/data.parquet")
procemDF.printSchema()
procemDF.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output for task 1:
# MAGIC
# MAGIC ```text
# MAGIC root
# MAGIC  |-- Time: long (nullable = true)
# MAGIC  |-- SolarPower: double (nullable = true)
# MAGIC  |-- WaterCooling01Power: double (nullable = true)
# MAGIC  |-- WaterCooling02Power: double (nullable = true)
# MAGIC  |-- VentilationPower: double (nullable = true)
# MAGIC  |-- Temperature: double (nullable = true)
# MAGIC  |-- WindSpeed: double (nullable = true)
# MAGIC  |-- Humidity: double (nullable = true)
# MAGIC  |-- ElectricityPrice: double (nullable = true)
# MAGIC
# MAGIC +-------------+----------+-------------------+-------------------+----------------+-----------+---------+--------+----------------+
# MAGIC |         Time|SolarPower|WaterCooling01Power|WaterCooling02Power|VentilationPower|Temperature|WindSpeed|Humidity|ElectricityPrice|
# MAGIC +-------------+----------+-------------------+-------------------+----------------+-----------+---------+--------+----------------+
# MAGIC |1716152400000|      NULL|               NULL|               NULL|            NULL|       NULL|     NULL|    NULL|           -0.31|
# MAGIC |1716152400168|      NULL|               NULL|               NULL|            NULL|    14.0357|  4.32466| 53.0894|            NULL|
# MAGIC |1716152400217|      NULL|               NULL|               NULL|    24744.613281|       NULL|     NULL|    NULL|            NULL|
# MAGIC |1716152400277|      NULL|        4370.453613|               NULL|            NULL|       NULL|     NULL|    NULL|            NULL|
# MAGIC |1716152400605|      NULL|               NULL|          49.490608|            NULL|       NULL|     NULL|    NULL|            NULL|
# MAGIC |1716152400906| -6.500515|               NULL|               NULL|            NULL|       NULL|     NULL|    NULL|            NULL|
# MAGIC |1716152401217|      NULL|               NULL|               NULL|    24749.710938|       NULL|     NULL|    NULL|            NULL|
# MAGIC |1716152401277|      NULL|        4358.939941|               NULL|            NULL|       NULL|     NULL|    NULL|            NULL|
# MAGIC |1716152401607|      NULL|               NULL|          51.819244|            NULL|       NULL|     NULL|    NULL|            NULL|
# MAGIC |1716152401906| -6.492893|               NULL|               NULL|            NULL|       NULL|     NULL|    NULL|            NULL|
# MAGIC +-------------+----------+-------------------+-------------------+----------------+-----------+---------+--------+----------------+
# MAGIC  ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2 - Calculate hourly averages
# MAGIC
# MAGIC ##### Background information:
# MAGIC
# MAGIC To get the hourly energy from the power: `hourly_energy (kWh) = average_power_for_the_hour (W) / 1000`
# MAGIC
# MAGIC The market price for electricity in Finland is given separately for each hour and does not change within the hour. Thus, there should be only one value for the price in each hour.
# MAGIC
# MAGIC The time in the ProCem data is given as UNIX timestamps in millisecond precision, i.e., how many milliseconds has passed since January 1, 1970.<br>
# MAGIC `1716152400000` corresponds to `Monday, May 20, 2024 00:00:00.000` in UTC+0300 timezone. Spark offers functions to do the conversion from the timestamps to a more human-readable format.
# MAGIC
# MAGIC As can be noticed from the data sample in task 1, the data contains a lot of NULL values. These NULL values mean that there is no measurement for that particular timestamp. Both the power and weather measurements are given roughly in one second intervals. Some measurements could be missing from the data, but those are not relevant for this exercise.
# MAGIC
# MAGIC
# MAGIC ##### Using the DataFrame from task 1:
# MAGIC
# MAGIC - calculate the electrical energy produced by the solar panels for each hour (in `kWh`)
# MAGIC - calculate the total combined electrical energy consumed by the water cooling and ventilation machinery for each hour (in `kWh`)
# MAGIC - determine the price of the electrical energy for each hour (in `€/MWh`)
# MAGIC - calculate the average temperature for each hour (in `°C`)
# MAGIC
# MAGIC Give the result as a DataFrame where one row contains the hour and the corresponding four values. Order the DataFrame by the hour with the earliest hour first.
# MAGIC
# MAGIC In the example output, the datetime representation for the hour is given in UTC+0300 timezone which was used in Finland (`Europe/Helsinki`) during May 2024.

# COMMAND ----------

from pyspark.sql.functions import from_utc_timestamp

# Convert Time column from milliseconds to timestamp and truncate to hour
hourlyDF = procemDF.withColumn(
    "Time",
    from_utc_timestamp(
        F.from_unixtime((F.col("Time") / 1000).cast("long"), 
                        "yyyy-MM-dd HH:00:00"), 
        "Europe/Helsinki"
    ).cast("timestamp")  # This parenthesis was missing
)

# Group by truncated Time and calculate metrics
hourlyDF = hourlyDF.groupBy("Time").agg(
    F.avg("Temperature").alias("AvgTemperature"),
    (F.avg("SolarPower") / 1000).alias("ProducedEnergy"),
    ((F.avg("WaterCooling01Power") + F.avg("WaterCooling02Power") + F.avg("VentilationPower")) / 1000).alias("ConsumedEnergy"),
    F.first("ElectricityPrice", ignorenulls=True).alias("Price")
).orderBy("Time")

hourlyDF.printSchema()
hourlyDF.show(8, False)



# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output for task 2:
# MAGIC
# MAGIC ```text
# MAGIC root
# MAGIC  |-- Time: timestamp (nullable = true)
# MAGIC  |-- AvgTemperature: double (nullable = true)
# MAGIC  |-- ProducedEnergy: double (nullable = true)
# MAGIC  |-- ConsumedEnergy: double (nullable = true)
# MAGIC  |-- Price: double (nullable = true)
# MAGIC
# MAGIC +-------------------+------------------+---------------------+------------------+-----+
# MAGIC |Time               |AvgTemperature    |ProducedEnergy       |ConsumedEnergy    |Price|
# MAGIC +-------------------+------------------+---------------------+------------------+-----+
# MAGIC |2024-05-20 00:00:00|13.754773048068902|-0.00583301298888889 |38.668148279083816|-0.31|
# MAGIC |2024-05-20 01:00:00|13.209065638888916|-0.005838116895833327|36.66061308699361 |-0.3 |
# MAGIC |2024-05-20 02:00:00|11.78702305555553 |-0.005843125166944462|37.91117283675028 |-0.1 |
# MAGIC |2024-05-20 03:00:00|10.510957036111114|-0.005829664534999985|35.09305738077221 |-0.03|
# MAGIC |2024-05-20 04:00:00|8.989454824999994 |0.0773709636641667   |36.59321714971756 |0.01 |
# MAGIC |2024-05-20 05:00:00|8.072131227777806 |0.8554100720902555   |33.8054992999987  |1.41 |
# MAGIC |2024-05-20 06:00:00|8.412301513888893 |2.3616457910869446   |54.32778184731544 |4.94 |
# MAGIC |2024-05-20 07:00:00|9.190588663888901 |11.20785037204834    |53.04797348656393 |10.44|
# MAGIC +-------------------+------------------+---------------------+------------------+-----+
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3 - Calculate daily prices
# MAGIC
# MAGIC ##### Background information:
# MAGIC
# MAGIC The energy that is considered to be bought from the electricity market is the difference between the consumed and produced energy.
# MAGIC
# MAGIC To get the hourly cost for the energy bought from the market: `hourly_cost (€) = hourly_energy_from_market (kWh) * electricity_price_for_hour (€/MWh) / 1000`
# MAGIC
# MAGIC Note, that any consumer buying electricity would also have to pay additional fees (taxes, transfer fees, etc.) that are not considered in this exercise.<br>
# MAGIC And that the given power consumption is only a part of the overall power consumption at Kampusareena.
# MAGIC
# MAGIC ##### Using the DataFrame from task 2 as a starting point:
# MAGIC
# MAGIC - calculate the average daily temperatures (in `°C`)
# MAGIC - calculate the average daily energy produced by the solar panels (in `kWh`)
# MAGIC - calculate the average daily energy consumed by the water cooling and ventilation machinery (in `kWh`)
# MAGIC - calculate the total daily price for the energy that was bought from the electricity market (in `€`)
# MAGIC
# MAGIC Give the result as a DataFrame where each row contains the date and the corresponding four values rounded to two decimals. Order the DataFrame by the date in chronological order.
# MAGIC
# MAGIC ##### Finally, calculate the total electricity price for the entire week.

# COMMAND ----------

from pyspark.sql import functions as F

# Add a "Date" column for daily grouping
hourlyDF = hourlyDF.withColumn("Date", F.to_date("Time"))

# # Calculate the hourly energy bought from the market and the hourly cost
# hourlyDF = hourlyDF.withColumn(
#     "EnergyBought",
#     F.when((F.col("ConsumedEnergy") - F.col("ProducedEnergy")) > 0, F.col("ConsumedEnergy") - F.col("ProducedEnergy")).otherwise(0)
# ).withColumn(
#     "HourlyCost",
#     F.col("EnergyBought") * F.col("Price") / 1000  # Convert €/MWh to €
# )

# Group by Date and calculate daily aggregates
dailyDF: DataFrame = hourlyDF.groupBy("Date").agg(
    F.round(F.avg("AvgTemperature"), 2).alias("AvgDailyTemperature"),
    F.round(F.avg("ProducedEnergy"), 2).alias("AvgDailyProducedEnergy"),
    F.round(F.avg("ConsumedEnergy"), 2).alias("AvgDailyConsumedEnergy"),
    F.round(F.sum((F.col("ConsumedEnergy") - F.col("ProducedEnergy")) * F.col("Price")/1000), 2).alias("DailyCost"),
    # F.round(F.sum("HourlyCost"), 2).alias("TotalDailyPrice")
).orderBy("Date")

dailyDF.show()
totalPrice: float = dailyDF.agg(F.sum("DailyCost")).collect()[0][0]
print(f"Total price: {totalPrice:.2f} EUR")


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output for task 3:
# MAGIC
# MAGIC ```text
# MAGIC +----------+-----------+--------------+--------------+---------+
# MAGIC |      Date|Temperature|ProducedEnergy|ConsumedEnergy|DailyCost|
# MAGIC +----------+-----------+--------------+--------------+---------+
# MAGIC |2024-05-20|      13.02|         373.9|       1084.76|     9.11|
# MAGIC |2024-05-21|      12.91|         369.7|        1154.5|    16.84|
# MAGIC |2024-05-22|      17.75|        355.15|       1708.37|    10.63|
# MAGIC |2024-05-23|      19.79|        360.76|       1948.03|      5.5|
# MAGIC |2024-05-24|      19.68|        258.41|       1978.22|    35.53|
# MAGIC |2024-05-25|      20.79|        294.36|       1533.66|     10.8|
# MAGIC |2024-05-26|       19.9|        265.03|       1264.05|     3.36|
# MAGIC +----------+-----------+--------------+--------------+---------+
# MAGIC
# MAGIC Total price: 91.77 EUR
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4 - Loading text data into an RDD
# MAGIC
# MAGIC In the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) the folder `exercises/ex3/wiki` contains texts from selected Wikipedia articles in raw text format.
# MAGIC
# MAGIC - Load all articles into a single RDD. Exclude all empty lines from the RDD.
# MAGIC - Count the total number of non-empty lines in the article collection.
# MAGIC - Pick the first 8 lines from the created RDD and print them out.

# COMMAND ----------

wikitextsRDD = spark.sparkContext.textFile("abfss://shared@tunics320f2024gen2.dfs.core.windows.net/exercises/ex3/wiki")
wikitextsRDD = wikitextsRDD.filter(lambda line: line.strip() != "")

numberOfLines: int = wikitextsRDD.count()
print(f"The number of lines with text: {numberOfLines}")

lines8 = wikitextsRDD.take(8)

print("============================================================")
# Print the first 60 characters of the first 8 lines
print(*[line[:60] for line in lines8], sep="\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output for task 4:
# MAGIC
# MAGIC ```text
# MAGIC The number of lines with text: 1113
# MAGIC ============================================================
# MAGIC Artificial intelligence
# MAGIC Artificial intelligence (AI), in its broadest sense, is inte
# MAGIC Some high-profile applications of AI include advanced web se
# MAGIC The various subfields of AI research are centered around par
# MAGIC Artificial intelligence was founded as an academic disciplin
# MAGIC Goals
# MAGIC The general problem of simulating (or creating) intelligence
# MAGIC Reasoning and problem-solving
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5 - Counting the number of words
# MAGIC
# MAGIC Using the RDD from task 4 as a starting point:
# MAGIC
# MAGIC - Create `wordsRdd` where each row contains one word with the following rules for words:
# MAGIC     - The word must have at least one character.
# MAGIC     - The word must not contain any numbers, i.e. the digits 0-9.
# MAGIC - Calculate the total number of words in the article collection using `wordsRDD`.
# MAGIC - Calculate the total number of distinct words in the article collection using the same criteria for the words.
# MAGIC
# MAGIC You can assume that words in the same line are separated from each other in the article collection by whitespace characters (` `).<br>
# MAGIC In this exercise you can ignore capitalization, parenthesis, and punctuation characters. I.e., `word`, `Word`, `WORD`, `word.`, `(word)`, and `word).` should all be considered as valid and distinct words for this exercise.

# COMMAND ----------

import re

wordsRDD = wikitextsRDD.flatMap(lambda line: line.split()) \
    .filter(lambda word: len(word) > 0 and not re.search(r'\d', word))  # Exclude words containing digits

numberOfWords = wordsRDD.count()
print(f"The total number of words not containing digits: {numberOfWords}")
numberOfDistinctWords = wordsRDD.distinct().count()
print(f"The total number of distinct words not containing digits: {numberOfDistinctWords}")


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output for task 5:
# MAGIC
# MAGIC ```text
# MAGIC The total number of words not containing digits: 44604
# MAGIC The total number of distinct words not containing digits: 9463
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 6 - Counting word occurrences with RDD
# MAGIC
# MAGIC - Using the article collection data, create a pair RDD, `wordCountRDD`, where each row contains a distinct word and the count for how many times the word can be found in the collection.
# MAGIC     - Use the same word criteria as in task 5, i.e., ignore words which contain digits.
# MAGIC
# MAGIC - Using the created pair RDD, find what is the most common 7-letter word that starts with an `s`, and what is its count in the collection.
# MAGIC     - You can modify the given code and find the word and its count separately if that seems easier for you.

# COMMAND ----------

import re
from pyspark import RDD

# Create the pair RDD where each row contains a distinct word and its count
wordCountRDD = (
    wikitextsRDD
    .flatMap(lambda line: line.split())
    .map(lambda word: re.sub(r'\d+', '', word))
    .filter(lambda word: len(word) > 0 and not any(char.isdigit() for char in word))
    .map(lambda word: (word, 1))
    .reduceByKey(lambda a, b: a + b)
)

print(f"First row in wordCountRDD: word: '{wordCountRDD.first()[0]}', count: {wordCountRDD.first()[1]}")
filteredWordsRDD = wordCountRDD.filter(lambda x: len(x[0]) == 7 and x[0].startswith('s'))
askedWord, wordCount = filteredWordsRDD.takeOrdered(1, key=lambda x: -x[1])[0]
print(f"The most common 7-letter word that starts with 's': {askedWord} (appears {wordCount} times)")



# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output for task 6:
# MAGIC
# MAGIC (the values for the first row depend on the code, and can be different than what is shown here)
# MAGIC
# MAGIC ```text
# MAGIC First row in wordCountRDD: word: 'sense,', count: 1
# MAGIC The most common 7-letter word that starts with 's': science (appears 93 times)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 7 - Handling text data with DataFrames
# MAGIC
# MAGIC In this task the same Wikipedia article collection as in the RDD tasks 4-6 is used.<br>
# MAGIC In the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) the folder `exercises/ex3/wiki` contains the texts from selected Wikipedia articles in raw text format.
# MAGIC
# MAGIC Scala supports Datasets, which are typed DataFrames, and the corresponding Scala task is about using them instead of RDDs to redo tasks 4 and 5.<br>
# MAGIC The Datasets are not supported by pyspark, thus this Python task is about using DataFrames instead.
# MAGIC
# MAGIC Part 1:
# MAGIC
# MAGIC - Do task 4 again, but this time using the higher level `DataFrame` instead of the low level `RDD`.
# MAGIC     - Load all articles into a single `DataFrame`. Exclude all empty lines from the DataFrame.
# MAGIC     - Count the total number of non-empty lines in the article collection.
# MAGIC     - Pick the first 8 lines from the created DataFrame and print them out.
# MAGIC
# MAGIC Part 2:
# MAGIC
# MAGIC - Do task 5 again, but this time using the higher level `DataFrame` instead of the low level `RDD`.
# MAGIC - Using the DataFrame from first part of this task as a starting point:
# MAGIC     - Create `wordsDF` where each row contains one word with the following rules for words:
# MAGIC         - The word must have at least one character.
# MAGIC         - The word must not contain any numbers, i.e. the digits 0-9.
# MAGIC     - Calculate the total number of words in the article collection using `wordsDF`.
# MAGIC     - Calculate the total number of distinct words in the article collection using the same criteria for the words.
# MAGIC
# MAGIC You can assume that words in the same line are separated from each other in the article collection by whitespace characters (` `).<br>
# MAGIC In this exercise you can ignore capitalization, parenthesis, and punctuation characters. I.e., `word`, `Word`, `word.`, and `(word)` should all be considered as valid and distinct words for this exercise.

# COMMAND ----------

from pyspark.sql import SparkSession
import re
spark = SparkSession.builder.appName("WikipediaTextProcessing").getOrCreate()

wikitextsDF = spark.read.text("abfss://shared@tunics320f2024gen2.dfs.core.windows.net/exercises/ex3/wiki")
wikitextsDF = wikitextsDF.filter(wikitextsDF["value"] != "")
linesInDF = wikitextsDF.count()
print(f"The number of lines with text: {linesInDF}")
first8Lines = wikitextsDF.select("value").rdd.flatMap(lambda x: x).take(8)

print("============================================================")
print(*[line[:60] for line in first8Lines], sep="\n")
wikitextsDF.show(8)

# COMMAND ----------

wordsRDD = wikitextsDF.rdd.flatMap(lambda line: line[0].split()) \
    .filter(lambda word: not any(char.isdigit() for char in word))  # Exclude words with digits
wordsDF = wordsRDD.map(lambda word: (word,)).toDF(["word"])
wordsInDF = wordsDF.count()
print(f"The total number of words (excluding words with digits): {wordsInDF}")

# Calculate the total number of distinct words
distinctWordsInDF = wordsDF.distinct().count()
print(f"The total number of distinct words (excluding words with digits): {distinctWordsInDF}")


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output for task 7:
# MAGIC
# MAGIC ```text
# MAGIC The number of lines with text: 1113
# MAGIC ============================================================
# MAGIC Artificial intelligence
# MAGIC Artificial intelligence (AI), in its broadest sense, is inte
# MAGIC Some high-profile applications of AI include advanced web se
# MAGIC The various subfields of AI research are centered around par
# MAGIC Artificial intelligence was founded as an academic disciplin
# MAGIC Goals
# MAGIC The general problem of simulating (or creating) intelligence
# MAGIC Reasoning and problem-solving
# MAGIC +--------------------+
# MAGIC |               value|
# MAGIC +--------------------+
# MAGIC |Artificial intell...|
# MAGIC |Artificial intell...|
# MAGIC |Some high-profile...|
# MAGIC |The various subfi...|
# MAGIC |Artificial intell...|
# MAGIC |               Goals|
# MAGIC |The general probl...|
# MAGIC |Reasoning and pro...|
# MAGIC +--------------------+
# MAGIC ```
# MAGIC
# MAGIC and
# MAGIC
# MAGIC ```text
# MAGIC The total number of words not containing digits: 44604
# MAGIC The total number of distinct words not containing digits: 9463
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 8 - Counting word occurrences with DataFrame
# MAGIC
# MAGIC Do task 6 again, but this time using the higher level `DataFrame` instead of the low level `RDD`.<br>
# MAGIC Also, this time you should use the given data class `WordCount` for the final result instead of `(str, int)` tuple like in task 6 with the RDD.
# MAGIC
# MAGIC - Using the article collection data, create a `DataFrame`, `wordCountDF`, where each row contains a distinct word and the count for how many times the word can be found in the collection.
# MAGIC     - Use the same word criteria as in task 7, i.e., ignore words which contain digits.
# MAGIC     - Have the column names in the DataFrame match the data class `WordCount`, i.e., `word` and `count`.
# MAGIC
# MAGIC - Using the created DataFrame, find what is the most common 7-letter word that starts with an `s`, and what is its count in the collection.
# MAGIC     - The straightforward way is to determine the word and its count together. However, you can determine the word and its count separately if you want.
# MAGIC     - Give the final result as an instance of the data class.

# COMMAND ----------

@dataclasses.dataclass(frozen=True)
class WordCount:
    word: str
    count: int

# COMMAND ----------

spark = SparkSession.builder.appName("WikipediaTextProcessing").getOrCreate()

wikitextsDF = spark.read.text("abfss://shared@tunics320f2024gen2.dfs.core.windows.net/exercises/ex3/wiki")
wikitextsDF = wikitextsDF.filter(wikitextsDF["value"] != "")
wordsDF = wikitextsDF.select(
    F.col("value").alias("line")
).rdd.flatMap(lambda line: line[0].split()) \
  .map(lambda word: re.sub(r'\d+', '', word)) \
  .filter(lambda word: len(word) > 0) \
  .map(lambda word: (word, 1))
wordCountDF = wordsDF.toDF(["word", "count"]) \
    .groupBy("word") \
    .sum("count") \
    .withColumnRenamed("sum(count)", "count")
print(f"First row in wordCountDF: word: '{wordCountDF.first()['word']}', count: {wordCountDF.first()['count']}")
filteredDF = wordCountDF.filter((F.col("word").rlike("^s")) & (F.length(F.col("word")) == 7))
mostCommonWordRow = filteredDF.orderBy(F.col("count").desc()).first()
if mostCommonWordRow:
    askedWordCount = WordCount(mostCommonWordRow['word'], mostCommonWordRow['count'])
    print(f"Type of askedWordCount: {type(askedWordCount)}")
    print(f"The most common 7-letter word that starts with 's': {askedWordCount.word} (appears {askedWordCount.count} times)")
else:
    print("No 7-letter word starting with 's' found in the collection.")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Example output for task 8:
# MAGIC
# MAGIC (the values for the first row depend on the code, and can be different than what is shown here)
# MAGIC
# MAGIC ```text
# MAGIC First row in wordCountDF: word: 'By', count: 12
# MAGIC Type of askedWordCount: <class '__main__.WordCount'>
# MAGIC The most common 7-letter word that starts with 's': science (appears 93 times)
# MAGIC ```
