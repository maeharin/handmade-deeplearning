from util import *
from mnist_loader import load_mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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


class TwoLayerNet:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.layer1 = None
        self.relu_layer = None
        self.layer2 = None
        self.softmax_with_loss_layer = None

    def train(self, x, t):
        # forward
        y = self._forward(x)

        # backward
        delta2 = self.softmax_with_loss_layer.backward(y, t)
        dx2 = self.layer2.backward(delta2)
        delta1 = self.relu_layer.backward(dx2)
        self.layer1.backward(delta1)

        # get new params
        w1, w2, b1, b2 = self.optimizer.update_params(
            w1=self.layer1.w,
            w2=self.layer2.w,
            b1=self.layer1.b,
            b2=self.layer2.b,
            dw1=self.layer1.dw,
            dw2=self.layer2.dw,
            db1=self.layer1.db,
            db2=self.layer2.db)

        # update params
        self.layer1.w = w1
        self.layer1.b = b1
        self.layer2.w = w2
        self.layer2.b = b2


    def predict(self, x):
        return self._forward(x)


    def _forward(self, x):
        u1 = self.layer1.forward(x)
        h1 = self.relu_layer.forward(u1)
        u2 = self.layer2.forward(h1)
        return self.softmax_with_loss_layer.forward(u2)


class Dense:
    def __init__(self, n_in, n_out):
        self.w = initialize_weights_he_normal(n_in, n_out)
        self.b = np.zeros(n_out).astype('float32')
        self.dw = None
        self.db = None
        self.x = None
        self.u = None

    def forward(self, x):
        self.x = x
        self.u = np.matmul(x, self.w) + self.b
        return self.u

    def backward(self, delta):
        self.dw = np.matmul(self.x.T, delta)
        self.db = np.sum(delta, axis=0) # sum for batch
        dx = np.matmul(delta, self.w.T)
        return dx


class SoftmaxWithLoss:
    def forward(self, x):
        return softmax(x)

    def backward(self, y, t):
        batch_size = y.shape[0]
        return y - t / batch_size # div for batch


class Relu:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return relu(self.x)

    def backward(self, dx):
        return dx * deriv_relu(self.x)


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


# build network
network = TwoLayerNet(optimizer=SGD(lr=0.01))
network.layer1 = Dense(784, 100)
network.relu_layer = Relu()
network.layer2 = Dense(100, 10)
network.softmax_with_loss_layer = SoftmaxWithLoss()

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
        network.train(x, t)

    # train loss, acc
    y_train = network.predict(x_train)
    train_loss = calc_loss(t_train, y_train)
    train_acc = accuracy_score(y_train.argmax(axis=1), t_train.argmax(axis=1))

    # validation loss, acc
    y_valid = network.predict(x_valid)
    valid_loss = calc_loss(t_valid, y_valid)
    valid_acc = accuracy_score(y_valid.argmax(axis=1), t_valid.argmax(axis=1))
    print("[epoch {}] train loss: {:.3f}, train acc: {:.3f}, valid loss: {:.3f}, valid acc: {:.3f}".format(
        epoch, train_loss, train_acc, valid_loss, valid_acc))


print("start predict..")
y_test = network.predict(x_test)
acc = accuracy_score(y_test.argmax(axis=1), t_test)
print("accuracy_score:{}".format(acc))
print("done")
