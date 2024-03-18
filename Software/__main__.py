import tensorflow as tf

from alexnet import build_alex_net
from resnet import build_resnet
from datasets import zenodo_ids, training_df, training_data

from functools import partial

def main():
    data = training_data(partial(training_df, zenodo_ids))

if __name__ == "__main__":
    main()
