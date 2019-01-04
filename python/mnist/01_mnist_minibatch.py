from util import *
from mnist_loader import load_mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#
# mini batch ver
#

np.random.seed(34)

def np_log(x):
    return np.log(np.clip(x, 1e-10, x))


# cross entropy error
def calc_loss(t, y):
    return - (t * np_log(y)).sum(axis=1).mean()


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
w1 = np.random.uniform(-np.sqrt(1.0/784), np.sqrt(1.0/784), (784, 100)).astype('float32')
b1 = np.zeros(100).astype('float32')
w2 = np.random.uniform(-np.sqrt(1.0/100), np.sqrt(1.0/100), (100, 10)).astype('float32')
b2 = np.zeros(10).astype('float32')

# learning rate
lr = 0.01

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

        w2 -= lr * dw2
        w1 -= lr * dw1
        b2 -= lr * db2
        b1 -= lr * db1

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
