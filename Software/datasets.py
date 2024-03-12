import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StringType

from os import listdir, name

# Might need to change these
FOLDER_PATH  : str = "/mnt/c/Users/Computing/Downloads/"
MAPPING_FILE : str = FOLDER_PATH + "gz2_filename_mapping.csv"
DATASET_FILE : str = FOLDER_PATH + "gz2_hart16.csv"
IMAGES_FOLDER: str = FOLDER_PATH + "images_gz2/images/"

spark: SparkSession = SparkSession.builder.getOrCreate()

mapping_names: DataFrame = spark.read.csv(path=MAPPING_FILE, header=True, inferSchema=True)
data_set     : DataFrame = spark.read.csv(path=DATASET_FILE, header=True, inferSchema=True)
file_names   : DataFrame = spark.createDataFrame(listdir(IMAGES_FOLDER), StringType())

print(file_names.head(10))