# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# For datasets see galaxy zoo 2:
#    labels: https://data.galaxyzoo.org/#section-7
#    images: https://zenodo.org/records/3565489#.Y3vFKS-l0eY,
#            https://portal.nersc.gov/project/dasrepo/self-supervised-learning-sdss/dataset.html
#
# The script assumes that you are using a UNIX based platform or windows subsystem for linux (WSL)
# gz2_filename_mapping.csv, gz2_hart16.csv and images_gz2 must all be found in your .../Downloads folder.
# NOTE: windows extract fails to extract gz2_hart16.csv correctly so use a different program like WinRAR
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# This script is used to make a dataset that contains both image data and labels.

import findspark

findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StringType, IntegerType
from pyspark.sql.functions import col, udf

from tensorflow._api.v2.io import read_file
from tensorflow._api.v2.image import decode_jpeg, decode_png, resize
from tensorflow._api.v2.v2 import Tensor, convert_to_tensor
from tensorflow._api.v2.dtypes import float32 as tf_float32

from keras.utils import Sequence

from pandas import DataFrame as pd_DataFrame
from numpy import array, ceil, flipud, int32 as np_int32, ndarray, rot90
from numpy.typing import NDArray
from numpy.random import randint

from sklearn.model_selection import train_test_split

from functools import partial
from os import listdir
from re import match, Match
from typing import Callable

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

DOWNLOADS_PATH            : str = "/home/luke/Downloads/"
MAPPING_FILE              : str = DOWNLOADS_PATH + "gz2_filename_mapping.csv"
DATASET_FILE              : str = DOWNLOADS_PATH + "gz2_hart16.csv"
ZENODO_IMAGES_FOLDER      : str = DOWNLOADS_PATH + "images_gz2/images/"
PREPROCESSED_IMAGES_FOLDER: str = DOWNLOADS_PATH + "images_gz2/preprocessed/"
DATASET_COLS              : list[str] = ["dr7objid", "sample", "gz2_class"]
DF_DROP_COLS              : list[str] = ["dr7objid", "objid"]
FILENAME_DROP_COLS        : list[str] = ["asset_id", "id", "sample"]

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

# [1]
@udf(returnType=StringType())
def remove_jpg_extension(name: str) -> str:
    """
    pyspark udf function for removing the .jpg extension from image files
    """
    return name.removesuffix(".jpg")


@udf(returnType=StringType())
def remove_png_extension(name: str) -> str:
    """
    pyspark udf function for removing the .jpg extension from image files
    """
    return name.removesuffix(".png")


@udf(returnType=StringType())
def jpg_to_png(name: str) -> str:
    """
    converts a column of images with .jpg extensions to .png extensions
    """
    return name.replace(".jpg", ".png")


@udf(returnType=IntegerType())
def classification(gz2_class: str) -> int:
    """
    pyspark udf that generates the correct class id (see `CLASSIFICATIONS`)
    """
    for stem in CLASSIFICATIONS.keys():
        m: Match | None = match(r"^" + stem, gz2_class)
        if m:
            return CLASSIFICATIONS[stem]
    return -1


# [2]
def zenodo_ids(session: SparkSession) -> DataFrame:
    """
    zenodo_id gets the object IDs from the extensions in
    /home/computing/Downloads/images_gz2/images/ once they have been
    downloaded from https://zenodo.org/records/3565489#.Y3vFKS-l0eY. Make sure
    both the images_gz2 and gz2_filename_mapping.csv are downloaded.
    """
    return session.createDataFrame(
        listdir(ZENODO_IMAGES_FOLDER), schema=StringType()
    ).withColumn("id", remove_jpg_extension(col("value")))


def preprocessed_ids(session: SparkSession) -> DataFrame:
    """
    preprocessed image are found in /home/computing/Downloads/images_gz2/images/
    and are loaded with this function. See README.md for command to download the
    preprocessed image set.
    """
    return session.createDataFrame(
        listdir(PREPROCESSED_IMAGES_FOLDER), schema=StringType()
    ).withColumn("id", remove_png_extension(col("value")))


def training_df(obj_func: Callable[[], DataFrame], png: bool) -> pd_DataFrame:
    """
    training df, takes a function that gives a dataframe of IDs and produces a
    list of existing image files names and classification numbers.
    """  
    spark = SparkSession.builder.getOrCreate()
    
    mapping_names: DataFrame = spark.read.csv(
        path=MAPPING_FILE, header=True, inferSchema=True
    )

    data_set: DataFrame = spark.read.csv(
        path=DATASET_FILE, header=True, inferSchema=True
    ).select(*DATASET_COLS)

    obj_ids: DataFrame = obj_func(spark)
    obj_ids = obj_ids.join(
        mapping_names, obj_ids["id"] == mapping_names["asset_id"], how="inner"
    ).drop(*FILENAME_DROP_COLS)

    df = obj_ids.join(data_set, obj_ids["objid"] == data_set["dr7objid"], how="inner")
    df = (
        df.drop(*DF_DROP_COLS)
        .withColumn("classification", classification(col("gz2_class")))
        .filter(col("classification") != -1)
        .sort(df["value"])
    )

    if png: df = df.withColumn("value", jpg_to_png(col("value")))
    
    pandas_df: pd_DataFrame = df.toPandas()
    
    spark.stop()
    del(spark)

    return pandas_df


def preprocess_image(image: Tensor) -> Tensor:
    """
    applies random flip and rotation on image
    """
    arr: NDArray = array(image)
    r: int = randint(0, 8)
    (rot, flip) = divmod(r, 2)
    if flip: arr = flipud(arr)
    arr = rot90(arr, rot)
    return convert_to_tensor(arr, dtype=tf_float32)
    

def load_raw_jpg(file_name: str, preprocessor: Callable[[Tensor], Tensor]) -> Tensor:
    """
    loads an image by name and performs a specified `preprocessor` function afterwards
    """
    return preprocessor(decode_jpeg(read_file(ZENODO_IMAGES_FOLDER + file_name)))


def load_preprocessed_png(file_name: str, preprocessor: Callable[[Tensor], Tensor]) -> Tensor:
    """
    loads a preprocessed image by name and performs a specified `preprocessor`
    function afterwards
    """
    return preprocessor(decode_png(read_file(PREPROCESSED_IMAGES_FOLDER + file_name)))


# [3]
class BatchGenerator(Sequence):
    def __init__(
        self,
        image_filenames: list[str],
        labels: list[int],
        batch_size: int,
        load_preprocess: partial[Tensor],
    ) -> None:
        self._image_filenames: list[str] = image_filenames
        self._labels: list[int] = labels
        self._batch_size: int = batch_size
        self._load_preprocess: Callable[[str], Tensor] = load_preprocess

    def __len__(self) -> np_int32:
        return (ceil(len(self._image_filenames) / float(self._batch_size))).astype(
            np_int32
        )

    def __getitem__(self, index):
        filenames_batch = self._image_filenames[
            batch_slice := slice(
                index * self._batch_size, (index + 1) * self._batch_size
            )
        ]
        labels_batch = self._labels[batch_slice]
        return array([*map(self._load_preprocess, filenames_batch)]) / 255., array(
            labels_batch
        )


def training_data(
    df: pd_DataFrame,
    preprocessed: bool,
    batch_size: int = 32,
    target_size: tuple[int, int] = (224, 224),
) -> tuple[BatchGenerator, BatchGenerator]:
    
    x_train, x_test, y_train, y_test = train_test_split(
        df["value"].tolist(),
        df["classification"].tolist(),
        train_size=0.8,
        shuffle=True,
    )

    # `load_preprocess` is higher order requiring a partial preprocessing method.
    # the inner partial function specifies the size of the preprocessed image.
    # load_preprocess: Callable[[str], Tensor] = lambda s: load_image(preprocess_image, s)
    load_preprocess: partial[Tensor] = (
        partial(load_preprocessed_png, preprocessor=preprocess_image)
        if preprocessed else
        partial(load_raw_jpg, preprocessor=preprocess_image)
    )
    return (
        BatchGenerator(x_train, y_train, batch_size, load_preprocess),
        BatchGenerator(x_test, y_test, batch_size, load_preprocess),
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
[1] # https://sparkbyexamples.com/pyspark/pyspark-udf-user-defined-function/
[2] # https://zenodo.org/records/3565489#.Y3vFKS-l0eY
[3] # https://gist.github.com/mrrajatgarg/6b55c86868d6376bb108ce7992595bb0#file-training_on_large_datasets_7-py
