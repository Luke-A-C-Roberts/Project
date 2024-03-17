import tensorflow as tf

from alexnet import build_alex_net
from resnet import build_resnet
from datasets import training_df

from datasets import *

def main():
    df = training_df()
    print(df)


if __name__ == "__main__":
    main()
