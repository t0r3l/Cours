#Job 1
#Create a function to filter clients => 18

import pyspark.sql.functions as f

def read_filter_join(spark):
    #Read
    clients = spark.read.option('header', True).csv("src/ressources/exo2/clients_bdd.csv")                                              
    cities = spark.read.option('header', True).csv("src/ressources/exo2/city_zipcode.csv")
    DF = [clients, cities]

    #Filter
    major_only = DF[0].where(f.col("age") >= 18)

    #Join with pySparkSQL paradigm
    # https://sparkbyexamples.com/pyspark/pyspark-join-explained-with-examples/
    major_and_cities  = major_only.join(cities, on="zip", how="inner")
    return major_and_cities 

def parquet_writter(df):
    df.write.mode('overwrite').parquet('data/exo2/clean')


def add_departements(df):
    departements_added  = df.withColumn(
        "departement",
        f.when(f.col("zip").substr(1, 2) == "20", 
        f.when(f.col("zip") <= "20190", "2A").otherwise("2B")
        ).otherwise(f.substring(f.col("zip"), 1, 2)))
    return departements_added 




