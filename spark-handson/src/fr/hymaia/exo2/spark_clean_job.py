
from pyspark.sql import SparkSession

#submodules
from src.fr.hymaia.exo2.clean.firstJob import *
from src.fr.hymaia.exo2.aggregate.secondJob import *


#Session creation
def main():
    spark = SparkSession.builder \
        .appName("exo_2") \
        .master("local[*]") \
        .getOrCreate()
    
#firstJob

    #First function
    #Read filter and join
    major_and_cities = read_filter_join(spark)
    
    #Write result in parquet format
    parquet_writter(major_and_cities)

    #Secound function
    #Departements
    departements_added = add_departements(major_and_cities)
   
    
    #Append dept column to 'data/exo2/output' thanks to overwrite write's option
    parquet_writter(departements_added)

#secondJob
    
    #Read clean
    population_per_dept(spark).show()
 

    #Ecriture du résultat au format csv
    #major_cities_and_departements.write.mode('overwrite').csv('`data/exo2/aggregate')
    
    

