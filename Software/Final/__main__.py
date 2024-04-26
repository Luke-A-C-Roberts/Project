from alexnet import build_alex_net
from resnet import build_resnet
from datasets import zenodo_ids, training_df, training_data

from tensorflow._api.v2.v2 import device
from tensorflow._api.v2.compat.v1 import ConfigProto, Session
from keras.models import Model
from keras.callbacks import History

from pandas import DataFrame

from functools import partial

TARGET_SIZE: tuple[int, int] = (224, 224)
BATCH_SIZE: int = 32
EPOCHS: int = 50
VERBOSITY: int = 1
TRAINING_EPOCH_STEPS: int = 3800
TESTING_EPOCH_STEPS: int = 950


def main() -> None:
    # [1]
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = Session(config=config)

    with device("/device:GPU:0"):
        training_generator, testing_generator = training_data(
            make_df=partial(training_df, zenodo_ids),
            batch_size=BATCH_SIZE,
            target_size=TARGET_SIZE,
        )
        # model = build_alex_net(4)
        model: Model = build_resnet(50, 4)

        # [2] NOTE: fit_generator is depreciated so just use fit
        history: History = model.fit(
            x=training_generator,
            steps_per_epoch=TRAINING_EPOCH_STEPS // BATCH_SIZE,
            validation_data=testing_generator,
            validation_steps=TESTING_EPOCH_STEPS // BATCH_SIZE,
            epochs=EPOCHS,
        )
        
        DataFrame(history.history).to_csv(
            "/home/computing/Project/Software/Final/resnet50_training_log.csv"
        )


if __name__ == "__main__":
    main()

[1] # https://www.linkedin.com/pulse/solving-out-memory-oom-errors-keras-tensorflow-running-wayne-cheng
[2] # https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
