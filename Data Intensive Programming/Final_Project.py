# Databricks notebook source
# MAGIC %md
# MAGIC # Data-Intensive Programming - Group assignment
# MAGIC
# MAGIC This is the **Python** version of the assignment. Switch to the Scala version, if you want to do the assignment in Scala.
# MAGIC
# MAGIC In all tasks, add your solutions to the cells following the task instructions. You are free to add new cells if you want.<br>
# MAGIC The example outputs, and some additional hints are given in a separate notebook in the same folder as this one.
# MAGIC
# MAGIC Don't forget to **submit your solutions to Moodle** once your group is finished with the assignment.
# MAGIC
# MAGIC ## Basic tasks (compulsory)
# MAGIC
# MAGIC There are in total nine basic tasks that every group must implement in order to have an accepted assignment.
# MAGIC
# MAGIC The basic task 1 is a separate task, and it deals with video game sales data. The task asks you to do some basic aggregation operations with Spark data frames.
# MAGIC
# MAGIC The other basic coding tasks (basic tasks 2-8) are all related and deal with data from [https://figshare.com/collections/Soccer_match_event_dataset/4415000/5](https://figshare.com/collections/Soccer_match_event_dataset/4415000/5) that contains information about events in [football](https://en.wikipedia.org/wiki/Association_football) matches in five European leagues during the season 2017-18. The tasks ask you to calculate the results of the matches based on the given data as well as do some further calculations. Special knowledge about football or the leagues is not required, and the task instructions should be sufficient in order to gain enough context for the tasks.
# MAGIC
# MAGIC Finally, the basic task 9 asks some information on your assignment working process.
# MAGIC
# MAGIC ## Advanced tasks (optional)
# MAGIC
# MAGIC There are in total of four advanced tasks that can be done to gain some course points. Despite the name, the advanced tasks may or may not be harder than the basic tasks.
# MAGIC
# MAGIC The advanced task 1 asks you to do all the basic tasks in an optimized way. It is possible that you gain some points from this without directly trying by just implementing the basic tasks efficiently. Logic errors and other issues that cause the basic tasks to give wrong results will be taken into account in the grading of the first advanced task. A maximum of 2 points will be given based on advanced task 1.
# MAGIC
# MAGIC The other three advanced tasks are separate tasks and their implementation does not affect the grade given for the advanced task 1.<br>
# MAGIC Only two of the three available tasks will be graded and each graded task can provide a maximum of 2 points to the total.<br>
# MAGIC If you attempt all three tasks, clearly mark which task you want to be used in the grading. Otherwise, the grader will randomly pick two of the tasks and ignore the third.
# MAGIC
# MAGIC Advanced task 2 continues with the football data and contains further questions that are done with the help of some additional data.<br>
# MAGIC Advanced task 3 deals with some image data and the questions are mostly related to the colors of the pixels in the images.<br>
# MAGIC Advanced task 4 asks you to do some classification related machine learning tasks with Spark.
# MAGIC
# MAGIC It is possible to gain partial points from the advanced tasks. I.e., if you have not completed the task fully but have implemented some part of the task, you might gain some appropriate portion of the points from the task. Logic errors, very inefficient solutions, and other issues will be taken into account in the task grading.
# MAGIC
# MAGIC ## Assignment grading
# MAGIC
# MAGIC Failing to do the basic tasks, means failing the assignment and thus also failing the course!<br>
# MAGIC "A close enough" solutions might be accepted => even if you fail to do some parts of the basic tasks, submit your work to Moodle.
# MAGIC
# MAGIC Accepted assignment submissions will be graded from 0 to 6 points.
# MAGIC
# MAGIC The maximum grade that can be achieved by doing only the basic tasks is 2/6 points (through advanced task 1).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Short summary
# MAGIC
# MAGIC ##### Minimum requirements (points: 0-2 out of maximum of 6):
# MAGIC
# MAGIC - All basic tasks implemented (at least in "a close enough" manner)
# MAGIC - Moodle submission for the group
# MAGIC
# MAGIC ##### For those aiming for higher points (0-6):
# MAGIC
# MAGIC - All basic tasks implemented
# MAGIC - Optimized solutions for the basic tasks (advanced task 1) (0-2 points)
# MAGIC - Two of the other three advanced tasks (2-4) implemented
# MAGIC     - Clearly marked which of the two tasks should be graded
# MAGIC     - Each graded advanced task will give 0-2 points
# MAGIC - Moodle submission for the group

# COMMAND ----------

# import statements for the entire notebook
# add anything that is required here

import re
from typing import Dict, List, Tuple

from pyspark.ml import Pipeline
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, sum as _sum, year, when, format_number, count, array_contains, month, hour, dayofweek, abs, udf, isnan, isnull
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType, DateType


# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 1 - Video game sales data
# MAGIC
# MAGIC The CSV file `assignment/sales/video_game_sales.csv` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) contains video game sales data (based on [https://www.kaggle.com/datasets/patkle/video-game-sales-data-from-vgchartzcom](https://www.kaggle.com/datasets/patkle/video-game-sales-data-from-vgchartzcom)).
# MAGIC
# MAGIC Load the data from the CSV file into a data frame. The column headers and the first few data lines should give sufficient information about the source dataset. The numbers in the sales columns are given in millions.
# MAGIC
# MAGIC Using the data, find answers to the following:
# MAGIC
# MAGIC - Which publisher has the highest total sales in video games in North America considering games released in years 2006-2015?
# MAGIC - How many titles in total for this publisher do not have sales data available for North America considering games released in years 2006-2015?
# MAGIC - Separating games released in different years and considering only this publisher and only games released in years 2006-2015, what are the total sales, in North America and globally, for each year?
# MAGIC     - I.e., what are the total sales (in North America and globally) for games released by this publisher in year 2006? And the same for year 2007? ...
# MAGIC

# COMMAND ----------

file = "abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/sales/video_game_sales.csv"

schema = StructType([
    StructField("title", StringType(), True),         
    StructField("publisher", StringType(), True),   
    StructField("developer", StringType(), True),   
    StructField("release_date", DateType(), True),   
    StructField("platform", StringType(), True),     
    StructField("total_sales", DoubleType(), True),  
    StructField("na_sales", DoubleType(), True),     
    StructField("japan_sales", DoubleType(), True),  
    StructField("pal_sales", DoubleType(), True),    
    StructField("other_sales", DoubleType(), True), 
    StructField("user_score", DoubleType(), True),   
    StructField("critic_score", DoubleType(), True)  
])

df = spark.read  \
  .option("header", "true") \
  .option("sep", "|") \
  .csv(file, schema = schema)

df_cleaned = (
    df.withColumn("release_year", year(col("release_date")))
    .filter((col("release_year") >= 2006) & (col("release_year") <= 2015))
)

df_cleaned = df_cleaned.cache()

na_sales_publisher = (
    df_cleaned.groupBy("publisher")
    .agg(_sum(col("na_sales")).alias("total_na_sales"))
    .orderBy(col("total_na_sales").desc())
    .limit(1)
)

bestNAPublisher = na_sales_publisher.first()["publisher"]

titlesWithMissingSalesData = (
    df_cleaned.filter((col("publisher") == bestNAPublisher) & (col("na_sales").isNull()))
    .count()
)

bestNAPublisherSales = (
    df_cleaned.filter(col("publisher") == bestNAPublisher)
    .groupBy("release_year")
    .agg(
        format_number(_sum(col("na_sales")), 2).alias("total_na_sales"),
        format_number(_sum(col("total_sales")), 2).alias("total_global_sales"),
    )
    .orderBy("release_year")
)

print(f"The publisher with the highest total video game sales in North America is: '{bestNAPublisher}'")
print(f"The number of titles with missing sales data for North America: {titlesWithMissingSalesData}")
print("Sales data for the publisher:")
bestNAPublisherSales.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 2 - Event data from football matches
# MAGIC
# MAGIC A parquet file in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) at folder `assignment/football/events.parquet` based on [https://figshare.com/collections/Soccer_match_event_dataset/4415000/5](https://figshare.com/collections/Soccer_match_event_dataset/4415000/5) contains information about events in [football](https://en.wikipedia.org/wiki/Association_football) matches during the season 2017-18 in five European top-level leagues: English Premier League, Italian Serie A, Spanish La Liga, German Bundesliga, and French Ligue 1.
# MAGIC
# MAGIC #### Background information
# MAGIC
# MAGIC In the considered leagues, a season is played in a double round-robin format where each team plays against all other teams twice. Once as a home team in their own stadium and once as an away team in the other team's stadium. A season usually starts in August and ends in May.
# MAGIC
# MAGIC Each league match consists of two halves of 45 minutes each. Each half runs continuously, meaning that the clock is not stopped when the ball is out of play. The referee of the match may add some additional time to each half based on game stoppages. \[[https://en.wikipedia.org/wiki/Association_football#90-minute_ordinary_time](https://en.wikipedia.org/wiki/Association_football#90-minute_ordinary_time)\]
# MAGIC
# MAGIC The team that scores more goals than their opponent wins the match.
# MAGIC
# MAGIC **Columns in the data**
# MAGIC
# MAGIC Each row in the given data represents an event in a specific match. An event can be, for example, a pass, a foul, a shot, or a save attempt.
# MAGIC
# MAGIC Simple explanations for the available columns. Not all of these will be needed in this assignment.
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | competition | string | The name of the competition |
# MAGIC | season | string | The season the match was played |
# MAGIC | matchId | integer | A unique id for the match |
# MAGIC | eventId | integer | A unique id for the event |
# MAGIC | homeTeam | string | The name of the home team |
# MAGIC | awayTeam | string | The name of the away team |
# MAGIC | event | string | The main category for the event |
# MAGIC | subEvent | string | The subcategory for the event |
# MAGIC | eventTeam | string | The name of the team that initiated the event |
# MAGIC | eventPlayerId | integer | The id for the player who initiated the event |
# MAGIC | eventPeriod | string | `1H` for events in the first half, `2H` for events in the second half |
# MAGIC | eventTime | double | The event time in seconds counted from the start of the half |
# MAGIC | tags | array of strings | The descriptions of the tags associated with the event |
# MAGIC | startPosition | struct | The event start position given in `x` and `y` coordinates in range \[0,100\] |
# MAGIC | enPosition | struct | The event end position given in `x` and `y` coordinates in range \[0,100\] |
# MAGIC
# MAGIC The used event categories can be seen from `assignment/football/metadata/eventid2name.csv`.<br>
# MAGIC And all available tag descriptions from `assignment/football/metadata/tags2name.csv`.<br>
# MAGIC You don't need to access these files in the assignment, but they can provide context for the following basic tasks that will use the event data.
# MAGIC
# MAGIC #### The task
# MAGIC
# MAGIC In this task you should load the data with all the rows into a data frame. This data frame object will then be used in the following basic tasks 3-8.

# COMMAND ----------

parquet_file_path = "abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/football/events.parquet"

eventDF: DataFrame = spark.read.parquet(parquet_file_path)
# display(eventDF)
display(eventDF.count())

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 3 - Calculate match results
# MAGIC
# MAGIC Create a match data frame for all the matches included in the event data frame created in basic task 2.
# MAGIC
# MAGIC The resulting data frame should contain one row for each match and include the following columns:
# MAGIC
# MAGIC | column name   | column type | description |
# MAGIC | ------------- | ----------- | ----------- |
# MAGIC | matchId       | integer     | A unique id for the match |
# MAGIC | competition   | string      | The name of the competition |
# MAGIC | season        | string      | The season the match was played |
# MAGIC | homeTeam      | string      | The name of the home team |
# MAGIC | awayTeam      | string      | The name of the away team |
# MAGIC | homeTeamGoals | integer     | The number of goals scored by the home team |
# MAGIC | awayTeamGoals | integer     | The number of goals scored by the away team |
# MAGIC
# MAGIC The number of goals scored for each team should be determined by the available event data.<br>
# MAGIC There are two events related to each goal:
# MAGIC
# MAGIC - One event for the player that scored the goal. This includes possible own goals.
# MAGIC - One event for the goalkeeper that tried to stop the goal.
# MAGIC
# MAGIC You need to choose which types of events you are counting.<br>
# MAGIC If you count both of the event types mentioned above, you will get double the amount of actual goals.

# COMMAND ----------

base_matches = eventDF.select(
    "matchId", "competition", "season", "homeTeam", "awayTeam"
).distinct()


goal_events = eventDF.filter(
    array_contains(col("tags"), "Goal")
)


goal_stats = goal_events.groupBy("matchId", "homeTeam", "awayTeam").agg(
    F.sum(
        F.when(
            (F.col("eventTeam") == F.col("homeTeam")),
            (F.when(array_contains(F.col("tags"), "Accurate") & array_contains(F.col("tags"), "Goal"), 1).otherwise(0)) +
            (F.when(array_contains(F.col("tags"), "Goal"), 1).otherwise(0) -
             2 * F.when(array_contains(F.col("tags"), "Accurate"), 1).otherwise(0))
        ).otherwise(0)
    ).alias("awayTeamGoals"),
    F.sum(
        F.when(
            (F.col("eventTeam") == F.col("awayTeam")),
            (F.when(array_contains(F.col("tags"), "Accurate") & array_contains(F.col("tags"), "Goal"), 1).otherwise(0)) +
            (F.when(array_contains(F.col("tags"), "Goal"), 1).otherwise(0) -
             2 * F.when(array_contains(F.col("tags"), "Accurate"), 1).otherwise(0))
        ).otherwise(0)
    ).alias("homeTeamGoals")
)


matchDF = base_matches.join(
    goal_stats,
    on=["matchId", "homeTeam", "awayTeam"],
    how="left"
).fillna({"homeTeamGoals": 0, "awayTeamGoals": 0})


total_matches = matchDF.count()


matches_without_goals = matchDF.filter(
    (F.col("homeTeamGoals") == 0) & (F.col("awayTeamGoals") == 0)
).count()


most_goals_in_single_game = matchDF.withColumn(
    "totalGoals", F.col("homeTeamGoals") + F.col("awayTeamGoals")
).agg(F.max("totalGoals")).first()[0]


total_goals = matchDF.withColumn(
    "totalGoals", F.col("homeTeamGoals") + F.col("awayTeamGoals")
).agg(F.sum("totalGoals")).first()[0]



print(f"Total number of matches: {total_matches}")
print(f"Matches without any goals: {matches_without_goals}")
print(f"Most goals in total in a single game: {most_goals_in_single_game}")
print(f"Total amount of goals: {total_goals}")

# specific_match_id = '2499806'
# Show specific match details with reordered columns
# specificMatchDF = matchDF.filter(F.col("matchId") == specific_match_id).select(
#     "matchId", "homeTeam", "awayTeam", "competition", "season", 
#     "homeTeamGoals", "awayTeamGoals"
# )
# specificMatchDF.show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 4 - Calculate team points in a season
# MAGIC
# MAGIC Create a season data frame that uses the match data frame from the basic task 3 and contains aggregated seasonal results and statistics for all the teams in all leagues. While the used dataset only includes data from a single season for each league, the code should be written such that it would work even if the data would include matches from multiple seasons for each league.
# MAGIC
# MAGIC ###### Game result determination
# MAGIC
# MAGIC - Team wins the match if they score more goals than their opponent.
# MAGIC - The match is considered a draw if both teams score equal amount of goals.
# MAGIC - Team loses the match if they score fewer goals than their opponent.
# MAGIC
# MAGIC ###### Match point determination
# MAGIC
# MAGIC - The winning team gains 3 points from the match.
# MAGIC - Both teams gain 1 point from a drawn match.
# MAGIC - The losing team does not gain any points from the match.
# MAGIC
# MAGIC The resulting data frame should contain one row for each team per league and season. It should include the following columns:
# MAGIC
# MAGIC | column name    | column type | description |
# MAGIC | -------------- | ----------- | ----------- |
# MAGIC | competition    | string      | The name of the competition |
# MAGIC | season         | string      | The season |
# MAGIC | team           | string      | The name of the team |
# MAGIC | games          | integer     | The number of games the team played in the given season |
# MAGIC | wins           | integer     | The number of wins the team had in the given season |
# MAGIC | draws          | integer     | The number of draws the team had in the given season |
# MAGIC | losses         | integer     | The number of losses the team had in the given season |
# MAGIC | goalsScored    | integer     | The total number of goals the team scored in the given season |
# MAGIC | goalsConceded  | integer     | The total number of goals scored against the team in the given season |
# MAGIC | points         | integer     | The total number of points gained by the team in the given season |

# COMMAND ----------

# Computing match results and assigning points for each team
homeResults = (
    matchDF
    .withColumn("team", F.col("homeTeam"))
    .withColumn("goalsScored", F.col("homeTeamGoals"))
    .withColumn("goalsConceded", F.col("awayTeamGoals"))
    .withColumn("wins", F.when(F.col("homeTeamGoals") > F.col("awayTeamGoals"), 1).otherwise(0))
    .withColumn("draws", F.when(F.col("homeTeamGoals") == F.col("awayTeamGoals"), 1).otherwise(0))
    .withColumn("losses", F.when(F.col("homeTeamGoals") < F.col("awayTeamGoals"), 1).otherwise(0))
    .withColumn("points", F.col("wins") * 3 + F.col("draws") * 1)
    .select("competition", "season", "team", "goalsScored", "goalsConceded", "wins", "draws", "losses", "points")
)

awayResults = (
    matchDF
    .withColumn("team", F.col("awayTeam"))
    .withColumn("goalsScored", F.col("awayTeamGoals"))
    .withColumn("goalsConceded", F.col("homeTeamGoals"))
    .withColumn("wins", F.when(F.col("awayTeamGoals") > F.col("homeTeamGoals"), 1).otherwise(0))
    .withColumn("draws", F.when(F.col("awayTeamGoals") == F.col("homeTeamGoals"), 1).otherwise(0))
    .withColumn("losses", F.when(F.col("awayTeamGoals") < F.col("homeTeamGoals"), 1).otherwise(0))
    .withColumn("points", F.col("wins") * 3 + F.col("draws") * 1)
    .select("competition", "season", "team", "goalsScored", "goalsConceded", "wins", "draws", "losses", "points")
)

# Combine home and away results
allResults = homeResults.union(awayResults)

# Aggregate results by team, competition, and season
seasonDF = (
    allResults
    .groupBy("competition", "season", "team")
    .agg(
        F.count("*").alias("games"),
        F.sum("wins").alias("wins"),
        F.sum("draws").alias("draws"),
        F.sum("losses").alias("losses"),
        F.sum("goalsScored").alias("goalsScored"),
        F.sum("goalsConceded").alias("goalsConceded"),
        F.sum("points").alias("points")
    )
    .orderBy("competition", "season", "points", ascending=[True, True, False])  # Sort by league, season, and points
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 5 - English Premier League table
# MAGIC
# MAGIC Using the season data frame from basic task 4 calculate the final league table for `English Premier League` in season `2017-2018`.
# MAGIC
# MAGIC The result should be given as data frame which is ordered by the team's classification for the season.
# MAGIC
# MAGIC A team is classified higher than the other team if one of the following is true:
# MAGIC
# MAGIC - The team has a higher number of total points than the other team
# MAGIC - The team has an equal number of points, but have a better goal difference than the other team
# MAGIC - The team has an equal number of points and goal difference, but have more goals scored in total than the other team
# MAGIC
# MAGIC Goal difference is the difference between the number of goals scored for and against the team.
# MAGIC
# MAGIC The resulting data frame should contain one row for each team.<br>
# MAGIC It should include the following columns (several columns renamed trying to match the [league table in Wikipedia](https://en.wikipedia.org/wiki/2017%E2%80%9318_Premier_League#League_table)):
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | Pos         | integer     | The classification of the team |
# MAGIC | Team        | string      | The name of the team |
# MAGIC | Pld         | integer     | The number of games played |
# MAGIC | W           | integer     | The number of wins |
# MAGIC | D           | integer     | The number of draws |
# MAGIC | L           | integer     | The number of losses |
# MAGIC | GF          | integer     | The total number of goals scored by the team |
# MAGIC | GA          | integer     | The total number of goals scored against the team |
# MAGIC | GD          | string      | The goal difference |
# MAGIC | Pts         | integer     | The total number of points gained by the team |
# MAGIC
# MAGIC The goal difference should be given as a string with an added `+` at the beginning if the difference is positive, similarly to the table in the linked Wikipedia article.

# COMMAND ----------

# Filter data for the English Premier League in season 2017-2018
eplSeasonDF = seasonDF.filter(
    (F.col("competition") == "English Premier League") & 
    (F.col("season") == "2017-2018")
)

# Calculate goal difference (GD)
eplSeasonDF = eplSeasonDF.withColumn("GD", F.col("goalsScored") - F.col("goalsConceded"))

# Format GD as a string with "+" if positive
eplSeasonDF = eplSeasonDF.withColumn(
    "GD_str", 
    F.when(F.col("GD") > 0, F.concat(F.lit("+"), F.col("GD").cast("string")))
    .otherwise(F.col("GD").cast("string"))
)

# Rank teams based on Points, Goal Difference, and Goals Scored
windowSpec = Window.orderBy(
    F.col("points").desc(), 
    F.col("GD").desc(), 
    F.col("goalsScored").desc()
)

# Add the position column by row number
eplRankedDF = eplSeasonDF.withColumn(
    "Pos", F.row_number().over(windowSpec)
)

# Select the required columns
finalEplTableDF = eplRankedDF.select(
    F.col("Pos"),
    F.col("team").alias("Team"),
    F.col("games").alias("Pld"),
    F.col("wins").alias("W"),
    F.col("draws").alias("D"),
    F.col("losses").alias("L"),
    F.col("goalsScored").alias("GF"),
    F.col("goalsConceded").alias("GA"),
    F.col("GD_str").alias("GD"),
    F.col("points").alias("Pts")
)

# Show the final English Premier League table
print("English Premier League Table for season 2017-2018")
finalEplTableDF.show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic task 6: Calculate the number of passes
# MAGIC
# MAGIC This task involves going back to the event data frame and counting the number of passes each team made in each match. A pass is considered successful if it is marked as `Accurate`.
# MAGIC
# MAGIC Using the event data frame from basic task 2, calculate the total number of passes as well as the total number of successful passes for each team in each match.<br>
# MAGIC The resulting data frame should contain one row for each team in each match, i.e., two rows for each match. It should include the following columns:
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | matchId     | integer     | A unique id for the match |
# MAGIC | competition | string      | The name of the competition |
# MAGIC | season      | string      | The season |
# MAGIC | team        | string      | The name of the team |
# MAGIC | totalPasses | integer     | The total number of passes the team attempted in the match |
# MAGIC | successfulPasses | integer | The total number of successful passes made by the team in the match |
# MAGIC
# MAGIC You can assume that each team had at least one pass attempt in each match they played.

# COMMAND ----------

# Filter for pass events and mark successful passes
passEventsDF = eventDF.filter(F.col("event") == "Pass")

# Count total passes and successful passes
passesDF = passEventsDF.withColumn(
    "successful", 
    F.when(F.array_contains(F.col("tags"), "Accurate"), 1).otherwise(0)
)

# Aggregate total passes and successful passes for each team in each match
aggregatedPassesDF = passesDF.groupby("matchId", "competition", "season", "eventTeam") \
    .agg(
        F.count("*").alias("totalPasses"),
        F.sum("successful").alias("successfulPasses")
    )

# Rename "eventTeam" to "team" for clarity
aggregatedPassesDF = aggregatedPassesDF.withColumnRenamed("eventTeam", "team")

# Fill missing successful passes with 0 where necessary
matchPassDF = aggregatedPassesDF.fillna({"successfulPasses": 0})

# Show the result
print("The total passes and successful passes for each team in each match")
matchPassDF.show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 7: Teams with the worst passes
# MAGIC
# MAGIC Using the match pass data frame from basic task 6 find the teams with the lowest average ratio for successful passes over the season `2017-2018` for each league.
# MAGIC
# MAGIC The ratio for successful passes over a single match is the number of successful passes divided by the number of total passes.<br>
# MAGIC The average ratio over the season is the average of the single match ratios.
# MAGIC
# MAGIC Give the result as a data frame that has one row for each league-team pair with the following columns:
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | competition | string      | The name of the competition |
# MAGIC | team        | string      | The name of the team |
# MAGIC | passSuccessRatio | double | The average ratio for successful passes over the season given as percentages rounded to two decimals |
# MAGIC
# MAGIC Order the data frame so that the team with the lowest ratio for passes is given first.

# COMMAND ----------

# Get only 2017-2018 season data
seasonPassDF = matchPassDF.filter(F.col("season") == "2017-2018")

# Compute the pass success ratio for each match
seasonRatDF = seasonPassDF.withColumn("passSuccessRatio", F.col("successfulPasses") / F.col("totalPasses")
)

# Compute the average pass success ratio for each team in each competition
teamAvgDF = seasonRatDF.groupBy("competition", "team").agg(
    F.avg("passSuccessRatio")
).withColumn(
    "avgPassSuccessRatio", F.round(F.col("avg(PassSuccessRatio)") * 100, 2)
)

# Find the minimum average pass success ratio for each competition
compMinDF = teamAvgDF.groupBy("competition").agg(
    F.min("avgPassSuccessRatio").alias("minPassSuccessRatio")
)

# Alias the DataFrames before joining
teamAvgDF_alias = teamAvgDF.alias("teamAvg")
compMinDF_alias = compMinDF.alias("compMin")

# Join to filter the teams with the minimum pass success ratio
lowestPassSuccessRatioUnorderedDF = teamAvgDF_alias.join(
    compMinDF_alias,
    (teamAvgDF_alias["competition"] == compMinDF_alias["competition"]) &
    (teamAvgDF_alias["avgPassSuccessRatio"] == compMinDF_alias["minPassSuccessRatio"])
).select(
    teamAvgDF_alias["competition"],
    teamAvgDF_alias["team"],
    teamAvgDF_alias["avgPassSuccessRatio"].alias("passSuccessRatio")
)

# Order data frame so that lowest ratio of passes is first
lowestPassSuccessRatioDF = lowestPassSuccessRatioUnorderedDF.orderBy("passSuccessRatio")

print("The teams with the lowest ratios for successful passes for each league in season 2017-2018:")
lowestPassSuccessRatioDF.show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic task 8: The best teams
# MAGIC
# MAGIC For this task the best teams are determined by having the highest point average per match.
# MAGIC
# MAGIC Using the data frames created in the previous tasks find the two best teams from each league in season `2017-2018` with their full statistics.
# MAGIC
# MAGIC Give the result as a data frame with the following columns:
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | Team        | string      | The name of the team |
# MAGIC | League      | string      | The name of the league |
# MAGIC | Pos         | integer     | The classification of the team within their league |
# MAGIC | Pld         | integer     | The number of games played |
# MAGIC | W           | integer     | The number of wins |
# MAGIC | D           | integer     | The number of draws |
# MAGIC | L           | integer     | The number of losses |
# MAGIC | GF          | integer     | The total number of goals scored by the team |
# MAGIC | GA          | integer     | The total number of goals scored against the team |
# MAGIC | GD          | string      | The goal difference |
# MAGIC | Pts         | integer     | The total number of points gained by the team |
# MAGIC | Avg         | double      | The average points per match gained by the team |
# MAGIC | PassRatio   | double      | The average ratio for successful passes over the season given as percentages rounded to two decimals |
# MAGIC
# MAGIC Order the data frame so that the team with the highest point average per match is given first.

# COMMAND ----------

# Get only 2017-2018 data
specificSeasonDF = seasonDF.filter(F.col("season") == "2017-2018")

# Compute the average points per match and goal difference (GD)
avgAndGoalDF = specificSeasonDF.withColumns(
    {"Avg": F.round(F.col("points") / F.col("games"), 2), "GD": F.col("goalsScored") - F.col("goalsConceded")}
)

# Format GD as a string with "+" if positive
avgAndGoalDF = avgAndGoalDF.withColumn(
    "GD_str", 
    F.when(F.col("GD") > 0, F.concat(F.lit("+"), F.col("GD").cast("string")))
    .otherwise(F.col("GD").cast("string"))
)

# Join with the average ratio of successful passes
withAvgPassDF = avgAndGoalDF.join(
    teamAvgDF, (avgAndGoalDF['team'] == teamAvgDF['team']) & (avgAndGoalDF['competition'] == teamAvgDF['competition'])
)

# Drop duplicate columns after the join
combinedDF = withAvgPassDF.drop(teamAvgDF["team"]).drop(teamAvgDF["competition"])

# Rank teams based on average points
windowSpec2 = Window.partitionBy("competition").orderBy(
    F.col("Avg").desc()
)

# Add the position column by row number
rankedCombinedDF = combinedDF.withColumn(
    "Pos", F.row_number().over(windowSpec2)
).orderBy(F.col("Pos"), F.col("Avg").desc())

# Select the required columns
bestDF = rankedCombinedDF.select(
    F.col("team").alias("Team"),
    F.col("competition").alias("League"),
    F.col("Pos"),
    F.col("games").alias("Pld"),
    F.col("wins").alias("W"),
    F.col("draws").alias("D"),
    F.col("losses").alias("L"),
    F.col("goalsScored").alias("GF"),
    F.col("goalsConceded").alias("GA"),
    F.col("GD_str").alias("GD"),
    F.col("points").alias("Pts"),
    F.col("Avg"),
    F.col("avgPassSuccessRatio").alias("PassRatio")
)

print("The top 2 teams for each league in season 2017-2018")
bestDF.show(10, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 9: General information
# MAGIC
# MAGIC Answer **briefly** to the following questions.
# MAGIC
# MAGIC Remember that using AI and collaborating with other students outside your group is allowed as long as the usage and collaboration is documented.<br>
# MAGIC However, every member of the group should have some contribution to the assignment work.
# MAGIC
# MAGIC - Who were your group members and their contributions to the work?
# MAGIC     - Solo groups can ignore this question.
# MAGIC
# MAGIC - Did you use AI tools while doing the assignment?
# MAGIC     - Which ones and how did they help?
# MAGIC
# MAGIC - Did you work with students outside your assignment group?
# MAGIC     - Who or which group? (only extensive collaboration need to reported)

# COMMAND ----------

# MAGIC %md
# MAGIC 1. The group members Walid Bayoud, Mubashir Khan, and Daniel Villagran. Walid was in charge of basic tasks 1-3, Mubashir was in charge of basic tasks 4-6, and Daniel was in charge of basic tasks 7-9. For the advanced tasks, we worked on them all together.
# MAGIC
# MAGIC 2. Yes, we used AI tools. We used ChatGPT to help with syntax and finding which functions to use for a task.
# MAGIC
# MAGIC 3. No, we did not work with students outside our assignment group.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced tasks
# MAGIC
# MAGIC The implementation of the basic tasks is compulsory for every group.
# MAGIC
# MAGIC Doing the following advanced tasks you can gain course points which can help in getting a better grade from the course.<br>
# MAGIC Partial solutions can give partial points.
# MAGIC
# MAGIC The advanced task 1 will be considered in the grading for every group based on their solutions for the basic tasks.
# MAGIC
# MAGIC The advanced tasks 2, 3, and 4 are separate tasks. The solutions used in these other advanced tasks do not affect the grading of advanced task 1. Instead, a good use of optimized methods can positively influence the grading of each specific task, while very non-optimized solutions can have a negative effect on the task grade.
# MAGIC
# MAGIC While you can attempt all three tasks (advanced tasks 2-4), only two of them will be graded and contribute towards the course grade.<br>
# MAGIC Mark in the following cell which tasks you want to be graded and which should be ignored.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### If you did the advanced tasks 2-4, mark here which of the two should be considered in grading:
# MAGIC
# MAGIC - Advanced task 2 should be graded: No
# MAGIC - Advanced task 3 should be graded: Yes
# MAGIC - Advanced task 4 should be graded: Yes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Task 1 - Optimized and correct solutions to the basic tasks (2 points)
# MAGIC
# MAGIC Use the tools Spark offers effectively and avoid unnecessary operations in the code for the basic tasks.
# MAGIC
# MAGIC A couple of things to consider (**not** even close to a complete list):
# MAGIC
# MAGIC - Consider using explicit schemas when dealing with CSV data sources.
# MAGIC - Consider only including those columns from a data source that are actually needed.
# MAGIC - Filter unnecessary rows whenever possible to get smaller datasets.
# MAGIC - Avoid collect or similar expensive operations for large datasets.
# MAGIC - Consider using explicit caching if some data frame is used repeatedly.
# MAGIC - Avoid unnecessary shuffling (for example sorting) operations.
# MAGIC - Avoid unnecessary actions (count, etc.) that are not needed for the task.
# MAGIC
# MAGIC In addition to the effectiveness of your solutions, the correctness of the solution logic will be taken into account when determining the grade for this advanced task 1.
# MAGIC "A close enough" solution with some logic fails might be enough to have an accepted group assignment, but those failings might lower the score for this task.
# MAGIC
# MAGIC It is okay to have your own test code that would fall into category of "ineffective usage" or "unnecessary operations" while doing the assignment tasks. However, for the final Moodle submission you should comment out or delete such code (and test that you have not broken anything when doing the final modifications).
# MAGIC
# MAGIC Note, that you should not do the basic tasks again for this additional task, but instead modify your basic task code with more efficient versions.
# MAGIC
# MAGIC You can create a text cell below this one and describe what optimizations you have done. This might help the grader to better recognize how skilled your work with the basic tasks has been.

# COMMAND ----------

# MAGIC %md
# MAGIC For this task, my teammates and I made sure to: 
# MAGIC - Avoid any unnecessary actions
# MAGIC - Avoid unnecessary shuffling/sorting
# MAGIC - Avoid any unnecessary expensive operations (collect...)
# MAGIC - Filter unnecessary rows whenever possible
# MAGIC - Drop duplicate rows after joining tables
# MAGIC - We used only the columns that were needed from the datasource
# MAGIC - We used explicit schemas when dealing with CSV data sources

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Task 2 - Further tasks with football data (2 points)
# MAGIC
# MAGIC This advanced task continues with football event data from the basic tasks. In addition, there are two further related datasets that are used in this task.
# MAGIC
# MAGIC A Parquet file at folder `assignment/football/matches.parquet` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) contains information about which players were involved on each match including information on the substitutions made during the match.
# MAGIC
# MAGIC Another Parquet file at folder `assignment/football/players.parquet` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) contains information about the player names, default roles when playing, and their birth areas.
# MAGIC
# MAGIC #### Columns in the additional data
# MAGIC
# MAGIC The match dataset (`assignment/football/matches.parquet`) has one row for each match and each row has the following columns:
# MAGIC
# MAGIC | column name  | column type | description |
# MAGIC | ------------ | ----------- | ----------- |
# MAGIC | matchId      | integer     | A unique id for the match |
# MAGIC | competition  | string      | The name of the league |
# MAGIC | season       | string      | The season the match was played |
# MAGIC | roundId      | integer     | A unique id for the round in the competition |
# MAGIC | gameWeek     | integer     | The gameWeek of the match |
# MAGIC | date         | date        | The date the match was played |
# MAGIC | status       | string      | The status of the match, `Played` if the match has been played |
# MAGIC | homeTeamData | struct      | The home team data, see the table below for the attributes in the struct |
# MAGIC | awayTeamData | struct      | The away team data, see the table below for the attributes in the struct |
# MAGIC | referees     | struct      | The referees for the match |
# MAGIC
# MAGIC Both team data columns have the following inner structure:
# MAGIC
# MAGIC | column name  | column type | description |
# MAGIC | ------------ | ----------- | ----------- |
# MAGIC | team         | string      | The name of the team |
# MAGIC | coachId      | integer     | A unique id for the coach of the team |
# MAGIC | lineup       | array of integers | A list of the player ids who start the match on the field for the team |
# MAGIC | bench        | array of integers | A list of the player ids who start the match on the bench, i.e., the reserve players for the team |
# MAGIC | substitution1 | struct     | The first substitution the team made in the match, see the table below for the attributes in the struct |
# MAGIC | substitution2 | struct     | The second substitution the team made in the match, see the table below for the attributes in the struct |
# MAGIC | substitution3 | struct     | The third substitution the team made in the match, see the table below for the attributes in the struct |
# MAGIC
# MAGIC Each substitution structs have the following inner structure:
# MAGIC | column name  | column type | description |
# MAGIC | ------------ | ----------- | ----------- |
# MAGIC | playerIn     | integer     | The id for the player who was substituted from the bench into the field, i.e., this player started playing after this substitution |
# MAGIC | playerOut    | integer     | The id for the player who was substituted from the field to the bench, i.e., this player stopped playing after this substitution |
# MAGIC | minute       | integer     | The minute from the start of the match the substitution was made.<br>Values of 45 or less indicate that the substitution was made in the first half of the match,<br>and values larger than 45 indicate that the substitution was made on the second half of the match. |
# MAGIC
# MAGIC The player dataset (`assignment/football/players.parquet`) has the following columns:
# MAGIC
# MAGIC | column name  | column type | description |
# MAGIC | ------------ | ----------- | ----------- |
# MAGIC | playerId     | integer     | A unique id for the player |
# MAGIC | firstName    | string      | The first name of the player |
# MAGIC | lastName     | string      | The last name of the player |
# MAGIC | birthArea    | string      | The birth area (nation or similar) of the player |
# MAGIC | role         | string      | The main role of the player, either `Goalkeeper`, `Defender`, `Midfielder`, or `Forward` |
# MAGIC | foot         | string      | The stronger foot of the player |
# MAGIC
# MAGIC #### Background information
# MAGIC
# MAGIC In a football match both teams have 11 players on the playing field or pitch at the start of the match. Each team also have some number of reserve players on the bench at the start of the match. The teams can make up to three substitution during the match where they switch one of the players on the field to a reserve player. (Currently, more substitutions are allowed, but at the time when the data is from, three substitutions were the maximum.) Any player starting the match as a reserve and who is not substituted to the field during the match does not play any minutes and are not considered involved in the match.
# MAGIC
# MAGIC For this task the length of each match should be estimated with the following procedure:
# MAGIC
# MAGIC - Only the additional time added to the second half of the match should be considered. I.e., the length of the first half is always considered to be 45 minutes.
# MAGIC - The length of the second half is to be considered as the last event of the half rounded upwards towards the nearest minute.
# MAGIC     - I.e., if the last event of the second half happens at 2845 seconds (=47.4 minutes) from the start of the half, the length of the half should be considered as 48 minutes. And thus, the full length of the entire match as 93 minutes.
# MAGIC
# MAGIC A personal plus-minus statistics for each player can be calculated using the following information:
# MAGIC
# MAGIC - If a goal was scored by the player's team when the player was on the field, `add 1`
# MAGIC - If a goal was scored by the opponent's team when the player was on the field, `subtract 1`
# MAGIC - If a goal was scored when the player was a reserve on the bench, `no change`
# MAGIC - For any event that is not a goal, or is in a match that the player was not involved in, `no change`
# MAGIC - Any substitutions is considered to be done at the start of the given minute.
# MAGIC     - I.e., if the player is substituted from the bench to the field at minute 80 (minute 35 on the second half), they were considered to be on the pitch from second 2100.0 on the 2nd half of the match.
# MAGIC - If a goal was scored in the additional time of the first half of the match, i.e., the goal event period is `1H` and event time is larger than 2700 seconds, some extra considerations should be taken into account:
# MAGIC     - If a player is substituted into the field at the beginning of the second half, `no change`
# MAGIC     - If a player is substituted off the field at the beginning of the second half, either `add 1` or `subtract 1` depending on team that scored the goal
# MAGIC     - Any player who is substituted into the field at minute 45 or later is only playing on the second half of the match.
# MAGIC     - Any player who is substituted off the field at minute 45 or later is considered to be playing the entire first half including the additional time.
# MAGIC
# MAGIC ### Tasks
# MAGIC
# MAGIC The target of the task is to use the football event data and the additional datasets to determine the following:
# MAGIC
# MAGIC - The players with the most total minutes played in season 2017-2018 for each player role
# MAGIC     - I.e., the player in Goalkeeper role who has played the longest time across all included leagues. And the same for the other player roles (Defender, Midfielder, and Forward)
# MAGIC     - Give the result as a data frame that has the following columns:
# MAGIC         - `role`: the player role
# MAGIC         - `player`: the full name of the player, i.e., the first name combined with the last name
# MAGIC         - `birthArea`: the birth area of the player
# MAGIC         - `minutes`: the total minutes the player played during season 2017-2018
# MAGIC - The players with higher than `+65` for the total plus-minus statistics in season 2017-2018
# MAGIC     - Give the result as a data frame that has the following columns:
# MAGIC         - `player`: the full name of the player, i.e., the first name combined with the last name
# MAGIC         - `birthArea`: the birth area of the player
# MAGIC         - `role`: the player role
# MAGIC         - `plusMinus`: the total plus-minus statistics for the player during season 2017-2018
# MAGIC
# MAGIC It is advisable to work towards the target results using several intermediate steps.

# COMMAND ----------

# mostMinutesDF: DataFrame = ???

# print("The players with the most minutes played in season 2017-2018 for each player role:")
# mostMinutesDF.show(truncate=False)

# COMMAND ----------

# topPlayers: DataFrame = ???

# print("The players with higher than +65 for the plus-minus statistics in season 2017-2018:")
# topPlayers.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Task 3 - Image data and pixel colors (2 points)
# MAGIC
# MAGIC This advanced task involves loading in PNG image data and complementing JSON metadata into Spark data structure. And then determining the colors of the pixels in the images, and finding the answers to several color related questions.
# MAGIC
# MAGIC The folder `assignment/openmoji/color` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) contains collection of PNG images from [OpenMoji](https://openmoji.org/) project.
# MAGIC
# MAGIC The JSON Lines formatted file `assignment/openmoji/openmoji.jsonl` contains metadata about the image collection. Only a portion of the images are included as source data for this task, so the metadata file contains also information about images not considered in this task.
# MAGIC
# MAGIC #### Data description and helper functions
# MAGIC
# MAGIC The image data considered in this task can be loaded into a Spark data frame using the `image` format: [https://spark.apache.org/docs/3.5.0/ml-datasource.html](https://spark.apache.org/docs/3.5.0/ml-datasource.html). The resulting data frame contains a single column which includes information about the filename, image size as well as the binary data representing the image itself. The Spark documentation page contains more detailed information about the structure of the column.
# MAGIC
# MAGIC Instead of using the images as source data for machine learning tasks, the binary image data is accessed directly in this task.<br>
# MAGIC You are given two helper functions to help in dealing with the binary data:
# MAGIC
# MAGIC - Function `toPixels` takes in the binary image data and the number channels used to represent each pixel.
# MAGIC     - In the case of the images used in this task, the number of channels match the number bytes used for each pixel.
# MAGIC     - As output the function returns an array of strings where each string is hexadecimal representation of a single pixel in the image.
# MAGIC - Function `toColorName` takes in a single pixel represented as hexadecimal string.
# MAGIC     - As output the function returns a string with the name of the basic color that most closely represents the pixel.
# MAGIC     - The function uses somewhat naive algorithm to determine the name of the color, and does not always give correct results.
# MAGIC     - Many of the pixels in this task have a lot of transparent pixels. Any such pixel is marked as the color `None` by the function.
# MAGIC
# MAGIC With the help of the given functions it is possible to transform the binary image data to an array of color names without using additional libraries or knowing much about image processing.
# MAGIC
# MAGIC The metadata file given in JSON Lines format can be loaded into a Spark data frame using the `json` format: [https://spark.apache.org/docs/3.5.0/sql-data-sources-json.html](https://spark.apache.org/docs/3.5.0/sql-data-sources-json.html). The attributes used in the JSON data are not described here, but are left for you to explore. The original regular JSON formatted file can be found at [https://github.com/hfg-gmuend/openmoji/blob/master/data/openmoji.json](https://github.com/hfg-gmuend/openmoji/blob/master/data/openmoji.json).
# MAGIC
# MAGIC ### Tasks
# MAGIC
# MAGIC The target of the task is to combine the image data with the JSON data, determine the image pixel colors, and the find the answers to the following questions:
# MAGIC
# MAGIC - Which four images have the most colored non-transparent pixels?
# MAGIC - Which five images have the lowest ratio of colored vs. transparent pixels?
# MAGIC - What are the three most common colors in the Finnish flag image (annotation: `flag: Finland`)?
# MAGIC     - And how many percentages of the colored pixels does each color have?
# MAGIC - How many images have their most common three colors as, `Blue`-`Yellow`-`Black`, in that order?
# MAGIC - Which five images have the most red pixels among the image group `activities`?
# MAGIC     - And how many red pixels do each of these images have?
# MAGIC
# MAGIC It might be advisable to test your work-in-progress code with a limited number of images before using the full image set.<br>
# MAGIC You are free to choose your own approach to the task: user defined functions with data frames, RDDs/Datasets, or combination of both.
# MAGIC
# MAGIC Note that the currently the Python helper functions do not exactly match the Scala versions, and thus the answers to the questions might not quite match the given example results in the example output notebook.

# COMMAND ----------

# separates binary image data to an array of hex strings that represent the pixels
# assumes 8-bit representation for each pixel (0x00 - 0xff)
# with `channels` attribute representing how many bytes is used for each pixel
def toPixels(data: bytes, channels: int) -> List[str]:
    return [
        "".join([
            f"{data[index+byte]:02X}"
            for byte in range(0, channels)
        ])
        for index in range(0, len(data), channels)
    ]

# COMMAND ----------

# naive implementation of picking the name of the pixel color based on the input hex representation of the pixel
# only works for OpenCV type CV_8U (mode=24) compatible input
def toColorName(hexString: str) -> str:
    # mapping of RGB values to basic color names
    colors: Dict[Tuple[int, int, int], str] = {
        (0, 0, 0):     "Black",  (0, 0, 128):     "Blue",   (0, 0, 255):     "Blue",
        (0, 128, 0):   "Green",  (0, 128, 128):   "Green",  (0, 128, 255):   "Blue",
        (0, 255, 0):   "Green",  (0, 255, 128):   "Green",  (0, 255, 255):   "Blue",
        (128, 0, 0):   "Red",    (128, 0, 128):   "Purple", (128, 0, 255):   "Purple",
        (128, 128, 0): "Green",  (128, 128, 128): "Gray",   (128, 128, 255): "Purple",
        (128, 255, 0): "Green",  (128, 255, 128): "Green",  (128, 255, 255): "Blue",
        (255, 0, 0):   "Red",    (255, 0, 128):   "Pink",   (255, 0, 255):   "Purple",
        (255, 128, 0): "Orange", (255, 128, 128): "Orange", (255, 128, 255): "Pink",
        (255, 255, 0): "Yellow", (255, 255, 128): "Yellow", (255, 255, 255): "White"
    }

    # helper function to round values of 0-255 to the nearest of 0, 128, or 255
    def roundColorValue(value: int) -> int:
        if value < 85:
            return 0
        if value < 170:
            return 128
        return 255

    validString: bool = re.match(r"[0-9a-fA-F]{8}", hexString) is not None
    if validString:
        # for OpenCV type CV_8U (mode=24) the expected order of bytes is BGRA
        blue: int = roundColorValue(int(hexString[0:2], 16))
        green: int = roundColorValue(int(hexString[2:4], 16))
        red: int = roundColorValue(int(hexString[4:6], 16))
        alpha: int = int(hexString[6:8], 16)

        if alpha < 128:
            return "None"  # any pixel with less than 50% opacity is considered as color "None"
        return colors[(red, green, blue)]

    return "None"  # any input that is not in valid format is considered as color "None"

# COMMAND ----------


# Load data
images = spark.read.format("image").option("dropInvalid", True).load("abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/openmoji/color")
images = images.select("image.origin", "image.data", "image.nChannels")

# Make given functions as UDFs
toPixels_udf = F.udf(lambda data, nChannels: toPixels(data, nChannels), ArrayType(StringType()))
toColorName_udf = F.udf(lambda hexes: [toColorName(h) for h in hexes], ArrayType(StringType()))

# Add a column of pixel hexcodes
withPixels = images.withColumn("pixels", toPixels_udf(F.col("data"), F.col("nChannels")))

# Add a column of color names
withColors = withPixels.withColumn("colorNames", toColorName_udf(F.col("pixels")))

# Add a column with hexcode for image
colorImages = withColors.withColumn("hexcode", F.regexp_extract(F.col("origin"), r".*/([^/]+)\.png$", 1)).select('colorNames', 'hexcode')
# colorImages.show()

# Load metadata
metadata = spark.read.format("json").load("abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/openmoji/metadata/openmoji.jsonl")
# metadata.printSchema()
# metadata.show()

# Add annotations and group to data
annotationsAndGroup = metadata.select("annotation", "group", "hexcode")
withAnnotationsAndGroup = annotationsAndGroup.join(colorImages, annotationsAndGroup.hexcode == colorImages.hexcode).select('annotation', 'group', 'colorNames')
withAnnotations = withAnnotationsAndGroup.select('annotation', 'colorNames')
# withAnnotations.show()

# COMMAND ----------

# Compute Transparent and Ratio columns
countTransparent = withAnnotations.withColumn(
    "Transparent", F.size(F.expr("filter(colorNames, x -> x = 'None')"))
).withColumns(
    {
        "RatioColorTrans": (F.size(F.col("colorNames"))-F.col("Transparent")) / F.col("Transparent"),
        "Color": F.size(F.col("colorNames")) - F.col("Transparent")
    }
).select("RatioColorTrans", "Color", "annotation")

# Find images with most colored pixels
orderedColor = countTransparent .orderBy(F.desc("Color"))
mostColoredImages = orderedColor.limit(4).select("annotation").collect()

# Find images with the lowest ratio of colored to transparent pixels
orderedRatioColorTrans = countTransparent .orderBy(F.asc("RatioColorTrans"))
leastColoredRatio = orderedRatioColorTrans.limit(5).select("annotation").collect()

# COMMAND ----------

# The annotations for the four images with the most colored non-transparent pixels
mostColoredPixels: List[str] = [row.annotation for row in mostColoredImages]

print("The annotations for the four images with the most colored non-transparent pixels:")
for image in mostColoredPixels:
    print(f"- {image}")
print("============================================================")


# The annotations for the five images having the lowest ratio of colored vs. transparent pixels
leastColoredPixels: List[str] = [row.annotation for row in leastColoredRatio]

print("The annotations for the five images having the lowest ratio of colored vs. transparent pixels:")
for image in leastColoredPixels:
    print(f"- {image}")

# COMMAND ----------

# Get 3 most common colors (with percentage) of Finnish Flag
finnishFlag = spark.createDataFrame([(s,) for s in withAnnotations.filter(F.col("annotation") == "flag: Finland").collect()[0].colorNames], ["color"]).filter(F.col("color") != "None")
finnishCommonColors = finnishFlag.groupBy("color").count().orderBy(F.desc("count")).withColumn("percent", F.round((F.col("count") / finnishFlag.count()) * 100, 2)).limit(3).select("color", "percent").collect()

# Turn list of colors into rows with single string for each pixel
expandColorNames = withAnnotations.withColumn("color", F.explode(F.col("colorNames"))).filter(F.col("color") != "None")

# Count the number of each color in an image
colorCounts = expandColorNames.groupBy("annotation", "color").count()

# Rank the colers in each image
windowSpec = Window.partitionBy("annotation").orderBy(F.desc("count"))
rankedColors = colorCounts.withColumn("rank", F.row_number().over(windowSpec))

# Get the 3 most common colors in an image
topColors = rankedColors.filter(F.col("rank") <= 3).groupBy("annotation").agg(F.collect_list("color").alias("top_3_colors"))

# Count the number of images that have their most common three colors as, Blue-Yellow-Black, in that exact order
blueYellowBlack = topColors.filter(F.col("top_3_colors") == F.array(F.lit("Blue"), F.lit("Yellow"), F.lit("Black")))

# COMMAND ----------

# The three most common colors in the Finnish flag image:
finnishFlagColors: List[str] = [row.color for row in finnishCommonColors]

# The percentages of the colored pixels for each common color in the Finnish flag image:
finnishColorShares: List[float] = [row.percent for row in finnishCommonColors]

print("The colors and their percentage shares in the image for the Finnish flag:")
for color, share in zip(finnishFlagColors, finnishColorShares):
    print(f"- color: {color}, share: {share}")
print("============================================================")


# The number of images that have their most common three colors as, Blue-Yellow-Black, in that exact order:
blueYellowBlackCount: int = blueYellowBlack.count()

print(f"The number of images that have, Blue-Yellow-Black, as the most common colors: {blueYellowBlackCount}")

# COMMAND ----------

# Get only the images in the activities group and only red pixels
activities = expandColorNames.filter((F.col("group") == "activities") & (F.col("color") == "Red"))

# Count the number of each color in an image
redCounts = activities.groupBy("annotation", "color").count()

# Get the 5 image with the most red pixels
topRed = redCounts.orderBy(F.desc("count")).limit(5).select("annotation", "count").collect()

# COMMAND ----------

# The annotations for the five images with the most red pixels among the image group activities:
redImageNames: List[str] = [row.annotation for row in topRed]

# The number of red pixels in the five images with the most red pixels among the image group activities:
redPixelAmounts: List[int] = [row['count'] for row in topRed]

print("The annotations and red pixel counts for the five images with the most red pixels among the image group 'activities':")
for color, pixel_count in zip(redImageNames, redPixelAmounts):
    print(f"- {color} (red pixels: {pixel_count})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Task 4 - Machine learning tasks (2 points)
# MAGIC
# MAGIC This advanced task involves experimenting with the classifiers provided by the Spark machine learning library. Time series data collected in the [ProCem](https://www.senecc.fi/projects/procem-2) research project is used as the training and test data. Similar data in a slightly different format was used in the first tasks of weekly exercise 3.
# MAGIC
# MAGIC The folder `assignment/energy/procem_13m.parquet` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) contains the time series data in Parquet format.
# MAGIC
# MAGIC #### Data description
# MAGIC
# MAGIC The dataset contains time series data from a period of 13 months (from the beginning of May 2023 to the end of May 2024). Each row contains the average of the measured values for a single minute. The following columns are included in the data:
# MAGIC
# MAGIC | column name        | column type   | description |
# MAGIC | ------------------ | ------------- | ----------- |
# MAGIC | time               | long          | The UNIX timestamp in second precision |
# MAGIC | temperature        | double        | The temperature measured by the weather station on top of Sähkötalo (`°C`) |
# MAGIC | humidity           | double        | The humidity measured by the weather station on top of Sähkötalo (`%`) |
# MAGIC | wind_speed         | double        | The wind speed measured by the weather station on top of Sähkötalo (`m/s`) |
# MAGIC | power_tenants      | double        | The total combined electricity power used by the tenants on Kampusareena (`W`) |
# MAGIC | power_maintenance  | double        | The total combined electricity power used by the building maintenance systems on Kampusareena (`W`) |
# MAGIC | power_solar_panels | double        | The total electricity power produced by the solar panels on Kampusareena (`W`) |
# MAGIC | electricity_price  | double        | The market price for electricity in Finland (`€/MWh`) |
# MAGIC
# MAGIC There are some missing values that need to be removed before using the data for training or testing. However, only the minimal amount of rows should be removed for each test case.
# MAGIC
# MAGIC ### Tasks
# MAGIC
# MAGIC - The main task is to train and test a machine learning model with [Random forest classifier](https://spark.apache.org/docs/3.5.0/ml-classification-regression.html#random-forests) in six different cases:
# MAGIC     - Predict the month (1-12) using the three weather measurements (temperature, humidity, and wind speed) as input
# MAGIC     - Predict the month (1-12) using the three power measurements (tenants, maintenance, and solar panels) as input
# MAGIC     - Predict the month (1-12) using all seven measurements (weather values, power values, and price) as input
# MAGIC     - Predict the hour of the day (0-23) using the three weather measurements (temperature, humidity, and wind speed) as input
# MAGIC     - Predict the hour of the day (0-23) using the three power measurements (tenants, maintenance, and solar panels) as input
# MAGIC     - Predict the hour of the day (0-23) using all seven measurements (weather values, power values, and price) as input
# MAGIC - For each of the six case you are asked to:
# MAGIC     1. Clean the source dataset from rows with missing values.
# MAGIC     2. Split the dataset into training and test parts.
# MAGIC     3. Train the ML model using a Random forest classifier with case-specific input and prediction.
# MAGIC     4. Evaluate the accuracy of the model with Spark built-in multiclass classification evaluator.
# MAGIC     5. Further evaluate the accuracy of the model with a custom build evaluator which should do the following:
# MAGIC         - calculate the percentage of correct predictions
# MAGIC             - this should correspond to the accuracy value from the built-in accuracy evaluator
# MAGIC         - calculate the percentage of predictions that were at most one away from the correct predictions taking into account the cyclic nature of the month and hour values:
# MAGIC             - if the correct month value was `5`, then acceptable predictions would be `4`, `5`, or `6`
# MAGIC             - if the correct month value was `1`, then acceptable predictions would be `12`, `1`, or `2`
# MAGIC             - if the correct month value was `12`, then acceptable predictions would be `11`, `12`, or `1`
# MAGIC         - calculate the percentage of predictions that were at most two away from the correct predictions taking into account the cyclic nature of the month and hour values:
# MAGIC             - if the correct month value was `5`, then acceptable predictions would be from `3` to `7`
# MAGIC             - if the correct month value was `1`, then acceptable predictions would be from `11` to `12` and from `1` to `3`
# MAGIC             - if the correct month value was `12`, then acceptable predictions would be from `10` to `12` and from `1` to `2`
# MAGIC         - calculate the average probability the model predicts for the correct value
# MAGIC             - the probabilities for a single prediction can be found from the `probability` column after the predictions have been made with the model
# MAGIC - As the final part of this advanced task, you are asked to do the same experiments (training+evaluation) with two further cases of your own choosing:
# MAGIC     - you can decide on the input columns yourself
# MAGIC     - you can decide the predicted attribute yourself
# MAGIC     - you can try some other classifier other than the random forest one if you want
# MAGIC
# MAGIC In all cases you are free to choose the training parameters as you wish.<br>
# MAGIC Note that it is advisable that while you are building your task code to only use a portion of the full 13-month dataset in the initial experiments.

# COMMAND ----------

# Initialize Spark session
spark = SparkSession.builder.appName("TimeSeriesClassification").getOrCreate()

# Load the dataset
data = spark.read.parquet("abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/energy/procem_13m.parquet")

# Define the classifier with default parameters
rf_classifier = RandomForestClassifier(featuresCol="features", seed=1)

# Function to train and evaluate models
def train_and_evaluate(case_data, input_cols, label_col, classifier, input_desc):

    # Splitting the data
    (trainingData, testData) = case_data.randomSplit([0.8, 0.2], seed=1)

    assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
    pipeline = Pipeline(stages=[assembler, classifier])
    
    model = pipeline.fit(trainingData)
    predictions = model.transform(testData)
    
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Training a '{classifier.__class__.__name__}' model to predict '{label_col}' based on input '{input_desc}'.")
    print(f"The accuracy of the model is {accuracy}")
    
    # Custom Evaluations
    correct = predictions.filter(predictions[label_col] == predictions.prediction).count() / testData.count()
    if "month" in case_data.columns:
        within_one = predictions.filter((abs(predictions[label_col] - predictions.prediction) <= 1) |
                                        (abs(predictions[label_col] - predictions.prediction) >= 11)).count() / testData.count()
        within_two = predictions.filter((abs(predictions[label_col] - predictions.prediction) <= 2) |
                                        (abs(predictions[label_col] - predictions.prediction) >= 10)).count() / testData.count()
    elif "hour" in case_data.columns:
        within_one = predictions.filter((abs(predictions[label_col] - predictions.prediction) <= 1) |
                                        (abs(predictions[label_col] - predictions.prediction) >= 23)).count() / testData.count()
        within_two = predictions.filter((abs(predictions[label_col] - predictions.prediction) <= 2) |
                                        (abs(predictions[label_col] - predictions.prediction) >= 22)).count() / testData.count()
    elif "dayofweek" in case_data.columns:
        within_one = predictions.filter((abs(predictions[label_col] - predictions.prediction) <= 1) |
                                        (abs(predictions[label_col] - predictions.prediction) >= 6)).count() / testData.count()
        within_two = predictions.filter((abs(predictions[label_col] - predictions.prediction) <= 2) |
                                        (abs(predictions[label_col] - predictions.prediction) >= 5)).count() / testData.count()
    
    # Extract the prediction probability element of the probability vector
    extract_prob = udf(lambda prob, pred: float(prob[pred]), DoubleType())
    correct_preds = predictions.filter(predictions[label_col] == predictions.prediction)
    correct_preds = correct_preds.withColumn("predicted_probability", extract_prob(col("probability"), col("prediction").cast("int")))
    # avg_prob = correct_preds.selectExpr("avg(predicted_probability)").first()[0]
    avg_prob = correct_preds.selectExpr("sum(predicted_probability)").withColumn("avg(predicted_probability)", col("sum(predicted_probability)") / testData.count()).select("avg(predicted_probability)").first()[0]

    return correct, within_one, within_two, avg_prob
    
    # print(f"Correct: {correct * 100:.2f}%")
    # print(f"Within one: {within_one * 100:.2f}%")
    # print(f"Within two: {within_two * 100:.2f}%")
    # print(f"Average probability for correct predictions: {avg_prob}")
    # print(10*'-')

# COMMAND ----------

rf_classifier.setLabelCol("month")
data_month = data.withColumn("month", month(col("time").cast("timestamp")))

# Case 1: Predict month using weather measurements
case1_data = data_month.dropna(subset=["temperature", "humidity", "wind_speed"])
correct1, within_one1, within_two1, avg_prob1 = train_and_evaluate(case1_data, ["temperature", "humidity", "wind_speed"], "month", rf_classifier, 'temperature,humidity,wind_speed')

# Case 2: Predict month using power measurements
case2_data = data_month.dropna(subset=["power_tenants", "power_maintenance", "power_solar_panels"])
correct2, within_one2, within_two2, avg_prob2 = train_and_evaluate(case2_data, ["power_tenants", "power_maintenance", "power_solar_panels"], "month", rf_classifier, 'power_tenants,power_maintenance,power_solar_panels')

# Case 3: Predict month using all measurements
case3_data = data_month.dropna(subset=["temperature", "humidity", "wind_speed", "power_tenants", "power_maintenance", "power_solar_panels", "electricity_price"])
correct3, within_one3, within_two3, avg_prob3 = train_and_evaluate(case3_data, ["temperature", "humidity", "wind_speed", "power_tenants", "power_maintenance", "power_solar_panels", "electricity_price"], "month", rf_classifier, 'temperature,humidity,wind_speed,power_tenants,power_maintenance,power_solar_panels,electricity_price')

rf_classifier.setLabelCol("hour")
data_hour = data.withColumn("hour", hour(col("time").cast("timestamp")))

# Case 4: Predict hour using weather measurements
case4_data = data_hour.dropna(subset=["temperature", "humidity", "wind_speed"])
correct4, within_one4, within_two4, avg_prob4 = train_and_evaluate(case4_data, ["temperature", "humidity", "wind_speed"], "hour", rf_classifier, 'temperature,humidity,wind_speed')

# Case 5: Predict hour using power measurements
case5_data = data_hour.dropna(subset=["power_tenants", "power_maintenance", "power_solar_panels"])
correct5, within_one5, within_two5, avg_prob5 = train_and_evaluate(case5_data, ["power_tenants", "power_maintenance", "power_solar_panels"], "hour", rf_classifier, 'power_tenants,power_maintenance,power_solar_panels')

# Case 6: Predict hour using all measurements
case6_data = data_hour.dropna(subset=["temperature", "humidity", "wind_speed", "power_tenants", "power_maintenance", "power_solar_panels", "electricity_price"])
correct6, within_one6, within_two6, avg_prob6 = train_and_evaluate(case6_data, ["temperature", "humidity", "wind_speed", "power_tenants", "power_maintenance", "power_solar_panels", "electricity_price"], "hour", rf_classifier, 'temperature,humidity,wind_speed,power_tenants,power_maintenance,power_solar_panels,electricity_price')

# COMMAND ----------

# Make and show dataframe with results
results = spark.createDataFrame([
    ["RandomForest", "temperature, humidity, wind_speed", "month", correct1, within_one1, within_two1, avg_prob1],
    ["RandomForest", "power_tenants, power_maintenance, power_solar_panels", "month", correct2, within_one2, within_two2, avg_prob2],
    ["RandomForest", "power_tenants, power_maintenance, power_solar_panels, electricity_price", "month", correct3, within_one3, within_two3, avg_prob3],
    ["RandomForest", "temperature, humidity, wind_speed", "hour", correct4, within_one4, within_two4, avg_prob4],
    ["RandomForest", "power_tenants, power_maintenance, power_solar_panels", "hour", correct5, within_one5, within_two5, avg_prob5],
    ["RandomForest", "power_tenants, power_maintenance, power_solar_panels, electricity_price", "hour", correct6, within_one6, within_two6, avg_prob6]
], ["classifier", "input", "label", "correct", "within_one", "within_two", "avg_prob"])
results_rounded = results.select(
    "classifier",
    "input",
    "label",
    F.round(F.col("correct") * 100, 2).alias("correct"),
    F.round(F.col("within_one") * 100, 2).alias("within_one"),
    F.round(F.col("within_two") * 100, 2).alias("within_two"),
    F.round(F.col("avg_prob"), 4).alias("avg_prob")

)

results_rounded.orderBy(F.col("label").desc(), F.col("correct").desc()).show(truncate=False)
# round make sure with next 2 cases

# COMMAND ----------

# Custom Experiments

# Case 7: Predict day of the week using power measurements with RandomForest
rf_classifier = RandomForestClassifier(labelCol="dayofweek", featuresCol="features")
data_dayofweek = data.withColumn("dayofweek", dayofweek(col("time").cast("timestamp")))
case7_data = data_dayofweek.dropna(subset=["power_tenants", "power_maintenance", "electricity_price"])
correct7, within_one7, within_two7, avg_prob7 = train_and_evaluate(
    case7_data,
    ["power_tenants", "power_maintenance", "electricity_price"],
    "dayofweek",
    rf_classifier,
    'power_tenants,power_maintenance,electricity_price'
)

# Case 8: Predict day of the week using power measurements with NaiveBayes
# Ensure all feature values are non-negative
case7_data_non_negative = case7_data.filter(
    (col("power_tenants") >= 0) &
    (col("power_maintenance") >= 0) &
    (col("electricity_price") >= 0)
)
nb_classifier = NaiveBayes(labelCol="dayofweek", featuresCol="features", smoothing=1.0)
correct8, within_one8, within_two8, avg_prob8 = train_and_evaluate(
    case7_data_non_negative,
    ["power_tenants", "power_maintenance", "electricity_price"],
    "dayofweek",
    nb_classifier,
    'power_tenants,power_maintenance,electricity_price'
)

results2 = spark.createDataFrame([
    ["RandomForest", "power_tenants, power_maintenance, electricity_price", "dayofweek", correct7, within_one7, within_two7, avg_prob7],
    ["NaiveBayes", "power_tenants, power_maintenance, electricity_price", "dayofweek", correct8, within_one8, within_two8, avg_prob8]
], ["classifier", "input", "label", "correct", "within_one", "within_two", "avg_prob"])

results_rounded2 = results2.select(
    "classifier",
    "input",
    "label",
    F.round(F.col("correct") * 100, 2).alias("correct"),
    F.round(F.col("within_one") * 100, 2).alias("within_one"),
    F.round(F.col("within_two") * 100, 2).alias("within_two"),
    F.round(F.col("avg_prob"), 4).alias("avg_prob")

)
results_rounded2.show(truncate=False)