#Job 2

import pyspark.sql.functions as f



def population_per_dept(spark):
    city_and_dept = spark.read.option('header', True).parquet("/home/t0r3l/Cours/Spark/spark-handson/`data/exo2/clean") \
    .select(f.col('city'), f.col('departement')) 
    
    .groupBy(f.col('departement')) \
    .count() \
    .orderBy(f.col("count").desc, f.col("departement").asc) \
    .WithColumnRename("count", "nb_people")
    return pop_per_dept


