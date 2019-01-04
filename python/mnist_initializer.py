import numpy as np
from util import *
from mnist_loader import load_mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#
# various initializer
#

np.random.seed(34)

def np_log(x):
    return np.log(np.clip(x, 1e-10, x))


# cross entropy error
def calc_loss(t, y):
    return - (t * np_log(y)).sum(axis=1).mean()


# Stochastic Gradient Descent
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update_params(self, w1, w2, b1, b2, dw1, dw2, db1, db2):
        w2 -= self.lr * dw2
        w1 -= self.lr * dw1
        b2 -= self.lr * db2
        b1 -= self.lr * db1
        return w1, w2, b1, b2


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.is_initialized = False
        self.v_w1 = None
        self.v_w2 = None
        self.v_b1 = None
        self.v_b2 = None

    def update_params(self, w1, w2, b1, b2, dw1, dw2, db1, db2):
        if not self.is_initialized:
            self.v_w1 = np.zeros_like(w1)
            self.v_w2 = np.zeros_like(w2)
            self.v_b1 = np.zeros_like(b1)
            self.v_b2 = np.zeros_like(b2)
            self.is_initialized = True

        self.v_w1 = self.momentum * self.v_w1 - self.lr * dw1
        w1 = w1 + self.v_w1

        self.v_w2 = self.momentum * self.v_w2 - self.lr * dw2
        w2 = w2 + self.v_w2

        self.v_b1 = self.momentum * self.v_b1 - self.lr * db1
        b1 = b1 + self.v_b1

        self.v_b2 = self.momentum * self.v_b2 - self.lr * db2
        b2 = b2 + self.v_b2

        return w1, w2, b1, b2


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.is_initialized = False
        self.h_w1 = None
        self.h_w2 = None
        self.h_b1 = None
        self.h_b2 = None

    def update_params(self, w1, w2, b1, b2, dw1, dw2, db1, db2):
        if not self.is_initialized:
            self.h_w1 = np.zeros_like(w1)
            self.h_w2 = np.zeros_like(w2)
            self.h_b1 = np.zeros_like(b1)
            self.h_b2 = np.zeros_like(b2)
            self.is_initialized = True

        # 勾配の二乗を加算していく
        self.h_w1 = self.h_w1 + dw1 * dw1
        w1 = w1 - self.lr * dw1 / (np.sqrt(self.h_w1) + 1e-7)

        self.h_w2 = self.h_w2 + dw2 * dw2
        w2 = w2 - self.lr * dw2 / (np.sqrt(self.h_w2) + 1e-7)

        self.h_b1 = self.h_b1 + db1 * db1
        b1 = b1 - self.lr * db1 / (np.sqrt(self.h_b1) + 1e-7)

        self.h_b2 = self.h_b2 + db2 * db2
        b2 = b2 - self.lr * db2 / (np.sqrt(self.h_b2) + 1e-7)

        return w1, w2, b1, b2


def initialize_weights_lucun_uniform(dim1, dim2):
    return np.random.uniform(low=-np.sqrt(1.0/dim1),
                             high=np.sqrt(1.0/dim1),
                             size=(dim1, dim2)).astype('float32')


def initialize_weights_xavier_uniform(dim1, dim2):
    return np.random.uniform(low=-np.sqrt(6.0/(dim1 + dim2)),
                             high=np.sqrt(6.0/(dim1 + dim2)),
                             size=(dim1, dim2)).astype('float32')


def initialize_weights_he_normal(dim1, dim2):
    return np.sqrt(2.0 / dim1) * np.random.normal(size=(dim1, dim2))


# load datas
x_train, t_train, x_test, t_test = load_mnist()

# normalize
x_train = x_train / 255
x_test = x_test / 255

# convert to one-hot
t_train = np.eye(10)[t_train]

# train valid split
x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train, test_size=0.2)
#x_train, t_train, x_test, t_test = x_train[0:101], t_train[0:101], x_test[0:101], t_test[0:101]


# init weights and biases
w1 = initialize_weights_he_normal(784, 100)
w2 = initialize_weights_he_normal(100, 10)
b1 = np.zeros(100).astype('float32')
b2 = np.zeros(10).astype('float32')

# learning rate
lr = 0.01
optimizer = SGD(lr=lr)
#optimizer = Momentum(lr=lr)
#optimizer = AdaGrad(lr=lr)

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

        w1, w2, b1, b2 = optimizer.update_params(w1,w2,b1,b2,dw1,dw2,db1,db2)

    # train loss, acc
    u1 = np.matmul(x_train, w1) + b1
    h1 = relu(u1)
    u2 = np.matmul(h1, w2) + b2
    y_train = softmax(u2)
    train_loss = calc_loss(t_train, y_train)
    train_acc = accuracy_score(y_train.argmax(axis=1), t_train.argmax(axis=1))

    # validation loss, acc
    u1 = np.matmul(x_valid, w1) + b1
    h1 = relu(u1)
    u2 = np.matmul(h1, w2) + b2
    y_valid = softmax(u2)
    valid_loss = calc_loss(t_valid, y_valid)
    valid_acc = accuracy_score(y_valid.argmax(axis=1), t_valid.argmax(axis=1))
    print("[epoch {}] train loss: {:.3f}, train acc: {:.3f}, valid loss: {:.3f}, valid acc: {:.3f}".format(
        epoch, train_loss, train_acc, valid_loss, valid_acc))


print("start predict..")
u1 = np.matmul(x_test, w1) + b1
h1 = relu(u1)
u2 = np.matmul(h1, w2) + b2
y_test = softmax(u2)
acc = accuracy_score(y_test.argmax(axis=1), t_test)
print("accuracy_score:{}".format(acc))
print("done")
