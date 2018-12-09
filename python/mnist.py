import numpy as np
from util import *
from mnist_loader import load_mnist

np.random.seed(34)
x_train, t_train, x_test, t_test = load_mnist()

# todo: 重み初期化

# todo: t_trainをk-foldに

# loop epoch
epochs = 10
for epoch in range(epochs):
    # todo: shffule
    x_train, t_train = x_train[:10], t_train[:10]
    # train
    for x, t in zip(x_train, t_train):
        print(x, t)
        # forward
        # input(784) -> hidden(32) -> output(10)

        # backprop

        # delta

        # update grad


# test
for x in x_test:
    # forward
    # determin
