from alexnet import build_alex_net
from resnet import build_resnet
from datasets import zenodo_ids, training_df, training_data

from tensorflow._api.v2.data import Dataset

from functools import partial


def main():
    data: Dataset = training_data(partial(training_df, zenodo_ids))
    print(data)

if __name__ == "__main__":
    main()
