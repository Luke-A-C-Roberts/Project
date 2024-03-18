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

# This script is used to make a dataset that contains both image data and labels.

import findspark

findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StringType, IntegerType
from pyspark.sql.functions import col, udf

from tensorflow._api.v2.data import Dataset
from tensorflow._api.v2.dtypes import string as tf_string, int32 as tf_int32, float32 as tf_float32
from tensorflow._api.v2.io import read_file
from tensorflow._api.v2.image import decode_jpeg, resize
from tensorflow._api.v2.v2 import constant, map_fn, Tensor, TensorSpec

from pandas import DataFrame as pd_DataFrame

from functools import partial
from os import listdir
from re import match
from typing import Callable

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Might need to change these
DOWNLOADS_PATH: str = "c:\\Users\\Computing\\Downloads\\" # "/mnt/c/Users/Computing/Downloads/"
MAPPING_FILE: str = DOWNLOADS_PATH + "gz2_filename_mapping.csv"
DATASET_FILE: str = DOWNLOADS_PATH + "gz2_hart16.csv"
ZENODO_IMAGES_FOLDER: str = DOWNLOADS_PATH + "images_gz2\\images\\"

DATASET_COLS: list[str] = ["dr7objid", "sample", "gz2_class"]
FILENAME_DROP_COLS: list[str] = ["asset_id", "id", "sample"]
DF_DROP_COLS: list[str] = ["dr7objid", "objid"]
CLASSIFICATIONS: dict[str, int] = {
    "Er" : 0,
    "Ei" : 0,
    "Ec" : 0,
    "Ser": 1,
    "Seb": 1,
    "Sen": 1,
    "Sa" : 2,
    "Sb" : 2,
    "Sc" : 2,
    "Sd" : 2,
    "SBa": 3,
    "SBb": 3,
    "SBc": 3,
    "SBd": 3,
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

spark = SparkSession.builder.getOrCreate()

# https://sparkbyexamples.com/pyspark/pyspark-udf-user-defined-function/
@udf(returnType=StringType())
def remove_jpg_extention(name: str) -> str:
    return name.removesuffix(".jpg")


@udf(returnType=IntegerType())
def classification(gz2_class: str) -> int:
    for stem in CLASSIFICATIONS.keys():
        m = match(r"^" + stem, gz2_class)
        if m: return CLASSIFICATIONS[stem]
    return -1


# https://zenodo.org/records/3565489#.Y3vFKS-l0eY
def zenodo_ids() -> DataFrame:
    """
    zenodo_id gets the object IDs from the extentions in
    /mnt/c/Users/Computing/Downloads/images_gz2/images/ once they have been
    downloaded from https://zenodo.org/records/3565489#.Y3vFKS-l0eY. Make sure
    both the images_gz2 and gz2_filename_mapping.csv are downloaded.
    """
    return spark.createDataFrame(
        listdir(ZENODO_IMAGES_FOLDER), schema=StringType()
    ).withColumn("id", remove_jpg_extention(col("value")))


def training_df(obj_func: Callable[[], DataFrame]) -> pd_DataFrame:
    """
    training df, takes an fuction that gives a dataframe of IDs and produces a
    list of existing image files names and classification numbers.
    """
    mapping_names: DataFrame = spark.read.csv(
        path=MAPPING_FILE, header=True, inferSchema=True
    )

    data_set: DataFrame = spark.read.csv(
        path=DATASET_FILE, header=True, inferSchema=True
    ).select(*DATASET_COLS)

    obj_ids: DataFrame = obj_func()
    obj_ids = obj_ids.join(
        mapping_names, obj_ids["id"] == mapping_names["asset_id"], how="inner"
    ).drop(*FILENAME_DROP_COLS)

    df = obj_ids.join(data_set, obj_ids["objid"] == data_set["dr7objid"], how="inner")
    df = (
        df
        .drop(*DF_DROP_COLS)
        .withColumn("classification", classification(col("gz2_class")))
        .filter(col("classification") != -1)
        .sort(df["value"])
    )

    return df.toPandas()


def preprocess_image(target_size: tuple[int, int], image: Tensor) -> Tensor:
    return resize(image, target_size)


def load_image(preprocessor: partial[Tensor], file_name: str) -> Tensor:
    return preprocessor(decode_jpeg(read_file(ZENODO_IMAGES_FOLDER + file_name)))


def training_data(make_df: partial[pd_DataFrame], target_size: tuple[int, int] = (224, 224)) -> tuple[Dataset, Dataset]:
    df: pd_DataFrame = make_df()
    labels = df["classification"].tolist()
    filenames = df["value"].tolist()

    load_preprocess = lambda s: load_image(partial(preprocess_image, target_size), s)
    images = map_fn(
        fn=load_preprocess,
        elems=constant(filenames, dtype=tf_string),
        fn_output_signature=TensorSpec(shape=[*target_size, 3])
    )

    return (
        Dataset.from_tensor_slices(images),
        Dataset.from_tensor_slices(constant(labels, dtype=tf_int32))
    )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
