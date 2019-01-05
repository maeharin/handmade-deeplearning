from util import *
from mnist_loader import load_mnist
import copy
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

    def update_params(self, w, b, dw, db):
        ww = w - self.lr * dw
        bb = b - self.lr * db
        return ww, bb


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.is_initialized = False
        self.v_w = None
        self.v_b = None

    def update_params(self, w, b, dw, db):
        if not self.is_initialized:
            self.v_w = np.zeros_like(w)
            self.v_b = np.zeros_like(b)
            self.is_initialized = True

        self.v_w = self.lr * dw + self.momentum * self.v_w
        self.v_b = self.lr * db + self.momentum * self.v_b
        ww = w - self.v_w
        bb = b - self.v_b
        return ww, bb


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.is_initialized = False
        self.h_w = None
        self.h_b = None

    def update_params(self, w, b, dw, db):
        if not self.is_initialized:
            self.h_w = np.zeros_like(w)
            self.h_b = np.zeros_like(b)
            self.is_initialized = True

        # 勾配の二乗を加算していく
        self.h_w = self.h_w + dw * dw
        self.h_b = self.h_b + db * db
        ww = w - self.lr * dw / (np.sqrt(self.h_w) + 1e-7)
        bb = b - self.lr * db / (np.sqrt(self.h_b) + 1e-7)
        return ww, bb


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


class MultiLayerNet:
    def __init__(self, optimizer):
        self.layers = []
        self.optimizer_prototype = optimizer

    def add(self, layer):
        if hasattr(layer, 'set_optimizer'):
            layer.set_optimizer(copy.copy(self.optimizer_prototype))
        self.layers.append(layer)

    def train(self, x, t):
        # forward
        y = self._forward(x)

        # backward
        o = None
        for i, layer in enumerate(self.layers[::-1]):
            if i == 0:  # last layer
                o = layer.backward(y, t)
            else:
                o = layer.backward(o)

    def predict(self, x):
        return self._forward(x)

    def _forward(self, x):
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y


class Dense:
    def __init__(self, n_in, n_out):
        self.optimizer = None
        self.w = initialize_weights_he_normal(n_in, n_out)
        self.b = np.zeros(n_out).astype('float32')
        self.x = None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def forward(self, x):
        self.x = x
        return np.matmul(x, self.w) + self.b

    def backward(self, delta):
        # for next layer
        dx = np.matmul(delta, self.w.T)

        # update this layer w and b
        dw = np.matmul(self.x.T, delta)
        db = np.sum(delta, axis=0) # sum for batch
        self.w, self.b = self.optimizer.update_params(self.w, self.b, dw, db)

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

# build network
network = MultiLayerNet(optimizer=Momentum(lr=0.01))
network.add(Dense(784, 100))
network.add(Relu())
network.add(Dense(100, 50))
network.add(Relu())
network.add(Dense(50, 10))
network.add(SoftmaxWithLoss())

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
