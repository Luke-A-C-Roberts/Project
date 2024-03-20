from alexnet import build_alex_net
from resnet import build_resnet
from datasets import zenodo_ids, training_df, training_data

from tensorflow._api.v2.v2 import device
from tensorflow._api.v2.compat.v1 import ConfigProto, Session
from sklearn.model_selection import train_test_split

from functools import partial

TARGET_SIZE: tuple[int, int] = (224, 224)
BATCH_SIZE: int = 32
EPOCHS: int = 50
VERBOSITY: int = 1
TRAINING_EPOCH_STEPS: int = 3800
TESTING_EPOCH_STEPS: int = 950


def main():
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
        # alexnet = build_alex_net()
        resnet = build_resnet(34, 4)

        # [2]
        resnet.fit_generator(
            generator=training_generator,
            steps_per_epoch=TRAINING_EPOCH_STEPS // BATCH_SIZE,
            epochs=EPOCHS,
            verbose=VERBOSITY,
            validation_data=testing_generator,
            validation_steps=TESTING_EPOCH_STEPS // BATCH_SIZE,
        )


if __name__ == "__main__":
    main()

[1] # https://www.linkedin.com/pulse/solving-out-memory-oom-errors-keras-tensorflow-running-wayne-cheng
[2] # https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
