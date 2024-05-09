# this script is standalone and meant to be used after model training
from datasets import *

from tensorflow._api.v2.v2 import device
from tensorflow._api.v2.config import list_physical_devices
from tensorflow.python.framework.config import set_memory_growth
from tensorflow.python.eager.context import PhysicalDevice

import keras
from functools import partial
from os import listdir

import pandas as pd

from sklearn.metrics import confusion_matrix

import findspark
findspark.init()

DATA_PATH = "/home/computing/Project/Software/Final/data/"

df = training_df(preprocessed_ids, True).iloc[:1000]
load_preprocess = partial(load_preprocessed_png, preprocessor=preprocess_image)

f = lambda s: "_model.keras" in s
model_names = [*filter(f, listdir(DATA_PATH))]
model_names = [name for name in model_names if name != "resnet152_model.keras"]

print(model_names)

gpu: PhysicalDevice = list_physical_devices('GPU')[0]
set_memory_growth(gpu, True)

cms = []

with device("/device:GPU:0"):
    for model_name in model_names:
        model = keras.saving.load_model(DATA_PATH + model_name, compile=True, safe_mode = False)
        generator = BatchGenerator(df["value"].to_list(), df["classification"].tolist(), 16, load_preprocess)
        out = model.predict(
            x=generator,
            batch_size = 16
        ).tolist()
        del(model)
        predictions = [*map(lambda subarr: subarr.index(max(subarr)), out)]
        cm = confusion_matrix(df["classification"].tolist(), predictions)
        cms.append(cm)


pd.DataFrame(data={"names": model_names, "confusion_mats": cms}).to_csv(DATA_PATH + "confusion_matrixes.csv")
