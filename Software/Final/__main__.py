from alexnet import build_alex_net
from resnet import build_resnet
from datasets import zenodo_ids, training_df, training_data

from tensorflow._api.v2.v2 import device
from tensorflow._api.v2.config import list_physical_devices
from tensorflow.python.framework.config import set_memory_growth
from tensorflow.python.eager.context import PhysicalDevice

from keras.api.models import Model
from keras.api.callbacks import History
from keras.api.backend import clear_session

from pandas import DataFrame

from collections import namedtuple
from functools import partial
from multiprocessing import Process
from typing import Callable

TARGET_SIZE: tuple[int, int] = (224, 224)
EPOCHS: int = 1
VERBOSITY: int = 1
TRAINING_EPOCH_STEPS: int = 3800
TESTING_EPOCH_STEPS: int = 950


# dictionary of models and lazy evaluated build_[model] functions
TrainingParams = namedtuple("TrainingParams", ["function", "batch_size"])
models: dict[str,Callable] = {
    # "alexnet"  : TrainingParams(partial(build_alex_net, 4), 32),
    # "resnet18" : TrainingParams(partial(build_resnet, 18,  4), 32),
    # "resnet34" : TrainingParams(partial(build_resnet, 34,  4), 32),
    # "resnet50" : TrainingParams(partial(build_resnet, 50,  4), 32),
    # "resnet101": TrainingParams(partial(build_resnet, 101, 4), 16),
    "resnet152": TrainingParams(partial(build_resnet, 152, 4), 16),
}

# [1]
def run_model(df: DataFrame, name: str, build: Callable, batch_size: int) -> None:
    print("\033[32m" + ("=" * 50),"building and training", name, ("=" * 50) + "\033[0m")
    
    gpu: PhysicalDevice = list_physical_devices('GPU')[0]
    set_memory_growth(gpu, True)

    with device("/device:GPU:0"):
        
        # training and validation generators
        training_generator, testing_generator = training_data(
            df=df,
            batch_size=batch_size,
            target_size=TARGET_SIZE,
        )

        # create models
        model: Model = build()

        # [2] NOTE: fit_generator is depreciated so just use fit
        history: History = model.fit(
            x=training_generator,
            steps_per_epoch=TRAINING_EPOCH_STEPS // batch_size,
            validation_data=testing_generator,
            validation_steps=TESTING_EPOCH_STEPS // batch_size,
            epochs=EPOCHS,
        )

        del(model)
        
        DataFrame(history.history).to_csv(
            f"/home/computing/Project/Software/Final/{name}_training_log.csv"
        )
    
    clear_session()


def process_model(df: DataFrame, name: str, build: Callable, batch_size: int) -> None:
    # [3]
    process = Process(target=run_model(df, name, build, batch_size))
    process.start()
    process.join()  


def main() -> None:
    df = training_df(zenodo_ids)
    for name, (build, batch_size) in models.items():
        process_model(df, name, build, batch_size)


if __name__ == "__main__":
    main()
