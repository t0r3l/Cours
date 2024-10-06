#Here, we create a main function which will be performed when instancing  "poetry run wordcount"
#wordcount is understood by poetry to perform as follow thanks to editing the pyproject.toml file as follow
#wordcount = "src.fr.hymaia.exo1.main:main

import pyspark.sql.functions as f
from pyspark.sql import SparkSession

#Here appName refers to name of spark session (here "first job" refering to worcount the below defined function
#to be applied to our dataframe)
#Master allows us to select number of ressources (*) means everything
#getOrCreate means that session will be overwritten if already exists or created if doesn't

def main():
    spark = SparkSession.builder \
        .appName("first job") \
        .master("local[*]") \
        .getOrCreate()
    #Here read.option methods are performed by our spark session to read file
    df = spark.read.option('header', True).csv("src/resources/exo1/data.csv")
    #wordcount defined under is there called to be applied on text column of df
    df_wordcount = wordcount(df, 'text')
    #.show est une action, ne doit pas servir en production
    df_wordcount.show()
    #Result is then partitioned by 'count' column then written in parquet format(compressed) to precised path
    #mode(overwrite) allows to overwirite outputed data each time we run the code
    df_wordcount.write.mode('overwrite').partitionBy('count').parquet('data/exo1/output')
    

def wordcount(df, col_name):
    #Here we define our job acting on precised column.
    #It's purpose is to split all data of a column based on ' ' separator
    #then to perform pipeline explode, groupBy and count.
    #It returns the consequent table
    return df.withColumn('word', f.explode(f.split(f.col(col_name), ' '))) \
        .groupBy('word') \
        .count()

