import numpy as np
from util import *
from mnist_loader import load_mnist

#
# mini batch ver
#

np.random.seed(34)
x_train, t_train, x_test, t_test = load_mnist()
#x_train, t_train, x_test, t_test = x_train[0:101], t_train[0:101], x_test[0:101], t_test[0:101]

# normalize
x_train = x_train / 255
x_test = x_test / 255

# init weights and biases
w1 = np.random.uniform(-np.sqrt(1.0/784), np.sqrt(1.0/784), (784, 100)).astype('float32')
b1 = np.zeros(100).astype('float32')
w2 = np.random.uniform(-np.sqrt(1.0/100), np.sqrt(1.0/100), (100, 10)).astype('float32')
b2 = np.zeros(10).astype('float32')

# convert to one-hot
t_train = np.eye(10)[t_train]

# loop epoch
epochs = 3
batch_size = 50
n_batch = len(x_train) // batch_size


print("start train...epochs: {}".format(epochs))
for epoch in range(epochs):
    x_train, t_train = shuffle(x_train, t_train)

    # train
    for i in range(n_batch):
        start = batch_size * i
        end = start + batch_size
        x = x_train[start:end]
        t = t_train[start:end]

        #
        # forward
        #
        u1 = np.matmul(x, w1) + b1
        h1 = relu(u1)
        u2 = np.matmul(h1, w2) + b2
        y = softmax(u2)

        #
        # backprop
        #

        # delta
        delta2 = y - t / batch_size # div for batch
        delta1 = np.matmul(delta2, w2.T) * deriv_relu(u1)

        # update grad
        dw2 = np.matmul(h1.T, delta2)
        dw1 = np.matmul(x.T, delta1)
        db2 = np.sum(delta2, axis=0) # sum for batch
        db1 = np.sum(delta1, axis=0) # sum for batch

        lr = 0.01
        w2 -= lr * dw2
        w1 -= lr * dw1
        b2 -= lr * db2
        b1 -= lr * db1

    print("epoch {} done.".format(epoch))


print("start predict..")
# test
predicts = []
for x, t in zip(x_test, t_test):
    # forward
    u1 = np.matmul(x, w1) + b1
    h1 = relu(u1)
    u2 = np.matmul(h1, w2) + b2
    y = softmax(u2)

    z = np.argmax(y)
    predicts.append([z, t])


test_count = len(predicts)
acc = [p for p in predicts if p[0] == p[1]]
acc_count = len(acc)
acc_ratio = acc_count / test_count
print("acc_count:{} / test_count:{} = {}".format(acc_count, test_count, acc_ratio))
print("done")
