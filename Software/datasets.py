# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# For datasets see galaxy zoo 2:
#    labels: https://data.galaxyzoo.org/#section-7 and images: https://zenodo.org/records/3565489#.Y3vFKS-l0eY
# The script assumes that windows subsystem for linux is available on your platform.
# gz2_filename_mapping.csv, gz2_hart16.csv and images_gz2 must all be found in your ...\Downloads folder.
# NOTE: windows extract fails to extract gz2_hart16.csv correctly so use a different program like WinRAR
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# This script is used to make a dataframe that contains both file names and labels.

import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StringType, IntegerType
from pyspark.sql.functions import col, udf

from pandas import DataFrame as pd_DataFrame

from os import listdir
from re import match

# Might need to change these
FOLDER_PATH  : str = "/mnt/c/Users/Computing/Downloads/"
MAPPING_FILE : str = FOLDER_PATH + "gz2_filename_mapping.csv"
DATASET_FILE : str = FOLDER_PATH + "gz2_hart16.csv"
IMAGES_FOLDER: str = FOLDER_PATH + "images_gz2/images/"

DATASET_COLS      : list[str] = ["dr7objid", "sample", "gz2_class"]
FILENAME_DROP_COLS: list[str] = ["asset_id", "id", "sample"]
DF_DROP_COLS      : list[str] = ["dr7objid", "objid"]

CLASSIFICATION_STEM: list[str] = ["Er", "Ei", "Ec", "Ser", "Seb", "Sen", "Sa", "Sb", "Sc", "Sd", "SBa", "SBb", "SBc", "SBd"]

# https://sparkbyexamples.com/pyspark/pyspark-udf-user-defined-function/
@udf(returnType=StringType())
def remove_jpg_extention(name: str) -> str:
    return name.removesuffix(".jpg")

@udf(returnType=IntegerType())
def classification(gz2_class: str) -> int:
    for stem in CLASSIFICATION_STEM:
        m = match(r"^" + stem, gz2_class)
        if m:
            return stem
    return "None"

def training_df() -> pd_DataFrame:
    spark: SparkSession = SparkSession.builder.getOrCreate()
    mapping_names: DataFrame = spark.read.csv(path=MAPPING_FILE, header=True, inferSchema=True)

    data_set: DataFrame = (
        spark
        .read
        .csv(path=DATASET_FILE, header=True, inferSchema=True)
        .select(*DATASET_COLS)
    )

    file_names: DataFrame = (
        spark
        .createDataFrame(listdir(IMAGES_FOLDER), schema=StringType())
        .withColumn("id", remove_jpg_extention(col("value")))
    )
    file_names = (
        file_names
        .join(mapping_names, file_names["id"] == mapping_names["asset_id"], how="inner")
        .drop(*FILENAME_DROP_COLS)
    )

    df = file_names.join(data_set, file_names["objid"] == data_set["dr7objid"], how="inner")
    df = (
        df
        .sort(df["value"])
        .drop(*DF_DROP_COLS)
        .withColumn("classification", classification(col("gz2_class")))
    )

    return df.toPandas()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

training_df().to_csv("/mnt/c/Users/Computing/Desktop/Project/Software/Final/result.csv")
