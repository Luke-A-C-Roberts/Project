# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# For datasets see galaxy zoo 2:
#    labels: https://data.galaxyzoo.org/#section-7
#    images: https://zenodo.org/records/3565489#.Y3vFKS-l0eY,
#            https://portal.nersc.gov/project/dasrepo/self-supervised-learning-sdss/dataset.html
#
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
from typing import Callable

# Might need to change these
DOWNLOADS_PATH      : str = "/mnt/c/Users/Computing/Downloads/"
MAPPING_FILE        : str = DOWNLOADS_PATH + "gz2_filename_mapping.csv"
DATASET_FILE        : str = DOWNLOADS_PATH + "gz2_hart16.csv"
ZENODO_IMAGES_FOLDER: str = DOWNLOADS_PATH + "images_gz2/images/"

DATASET_COLS       : list[str] = ["dr7objid", "sample", "gz2_class"]
FILENAME_DROP_COLS : list[str] = ["asset_id", "id", "sample"]
DF_DROP_COLS       : list[str] = ["dr7objid", "objid"]
CLASSIFICATION_STEM: list[str] = ["Er", "Ei", "Ec", "Ser", "Seb", "Sen", "Sa", "Sb", "Sc", "Sd", "SBa", "SBb", "SBc", "SBd"]
CLASSIFICATION_INT : list[int] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]

spark: SparkSession = SparkSession.builder.getOrCreate()

# https://sparkbyexamples.com/pyspark/pyspark-udf-user-defined-function/
@udf(returnType=StringType())
def remove_jpg_extention(name: str) -> str:
    return name.removesuffix(".jpg")


@udf(returnType=IntegerType())
def classification(gz2_class: str) -> str:
    for stem in CLASSIFICATION_STEM:
        m = match(r"^" + stem, gz2_class)
        if m:
            return CLASSIFICATION_INT[CLASSIFICATION_STEM.index(stem)]
    return -1


# https://zenodo.org/records/3565489#.Y3vFKS-l0eY
def zenodo_ids() -> DataFrame:
    """
    zenodo_id gets the object IDs from the extentions in /mnt/c/Users/Computing/Downloads/images_gz2/images/ once they have been downloaded from https://zenodo.org/records/3565489#.Y3vFKS-l0eY. Make sure both the images_gz2 and gz2_filename_mapping.csv are downloaded. 
    """
    return (
        spark
        .createDataFrame(listdir(ZENODO_IMAGES_FOLDER), schema=StringType())
        .withColumn("id", remove_jpg_extention(col("value")))
    )


def training_df(obj_ids: Callable[[None], DataFrame]) -> pd_DataFrame:
    """
    training df, takes an fuction that gives a dataframe of IDs and produces a list of existing image files names and classification numbers.
    """
    mapping_names: DataFrame = spark.read.csv(path=MAPPING_FILE, header=True, inferSchema=True)

    data_set: DataFrame = (
        spark
        .read
        .csv(path=DATASET_FILE, header=True, inferSchema=True)
        .select(*DATASET_COLS)
    )

    obj_ids: DataFrame = obj_ids()
    obj_ids = (
        obj_ids()
        .join(mapping_names, obj_ids["id"] == mapping_names["asset_id"], how="inner")
        .drop(*FILENAME_DROP_COLS)
    )

    df = obj_ids.join(data_set, obj_ids["objid"] == data_set["dr7objid"], how="inner")
    df = (
        df
        .sort(df["value"])
        .drop(*DF_DROP_COLS)
        .withColumn("classification", classification(col("gz2_class")))
        .filter(col("classification") != -1)
    )

    return df.toPandas()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# training_df(zenodo_ids).to_csv("/mnt/c/Users/Computing/Desktop/Project/Software/Final/result.csv")
