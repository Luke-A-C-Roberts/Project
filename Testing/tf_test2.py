import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from keras.datasets import cifar10

import alex_net


# Main ########################################################################


def main() -> None:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    format = lambda dataset: dataset.astype("float32") / 255.0
    x_train, x_test = [*map(format, [x_train, x_test])]
    alexnet = build_alex_net(x_train.shape())


if __name__ == "__main__":
    main()
