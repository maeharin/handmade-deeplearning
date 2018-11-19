import numpy as np

np.random.seed(34)

def sigmoid(x):
    #return np.tanh(x * 0.5) * 0.5 + 0.5
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(x, 0)

def deriv_relu(x):
    return (x > 0).astype(x.dtype)

# XORデータセット
x_train_xor = np.array([
    [0, 1], [1, 0], [0, 0], [1, 1]
])
t_train_xor = np.array([
    [1], [1], [0], [0]
])
x_valid_xor, t_valid_xor = x_train_xor, t_train_xor

# 重み (入力層の次元数: 2, 隠れ層の次元数: 8, 出力層の次元数: 1)
W1 = np.random.uniform(low=-0.08, high=0.08, size=(2, 8)).astype('float32')
b1 = np.zeros(8).astype('float32')
W2 = np.random.uniform(low=-0.08, high=0.08, size=(8, 1)).astype('float32')
b2 = np.zeros(1).astype('float32')

# logの中身が 0になるのを防ぐ
def np_log(x):
    return np.log(np.clip(x, 1e-10, x))

def train_xor(x, t, eps):
    """
    :param x: np.ndarray, 入力データ, shape=(batch_size, 入力層の次元数)
    :param t: np.ndarray, 教師ラベル, shape=(batch_size, 出力層の次元数)
    :param eps: float, 学習率
    """
    global W1, b1, W2, b2

    batch_size = x.shape[0]

    # 順伝播
    u1 = np.matmul(x, W1) + b1 # shape: (batch_size, 隠れ層の次元数)
    h1 = relu(u1)

    u2 = np.matmul(h1, W2) + b2 # shape: (batch_size, 出力層の次元数)
    y = sigmoid(u2)

    # 誤差の計算
    cost = (- t * np_log(y) - (1 - t) * np_log(1 - y)).mean()

    # 逆伝播
    delta_2 = y - t # shape: (batch_size, 出力層の次元数)
    delta_1 = deriv_relu(u1) * np.matmul(delta_2, W2.T) # shape: (batch_size, 隠れ層の次元数)

    # 勾配の計算
    dW1 = np.matmul(x.T, delta_1) / batch_size # shape: (入力層の次元数, 隠れ層の次元数)
    db1 = np.matmul(np.ones(batch_size), delta_1) / batch_size # shape: (隠れ層の次元数,)

    dW2 = np.matmul(h1.T, delta_2) / batch_size # shape: (隠れ層の次元数, 出力層の次元数)
    db2 = np.matmul(np.ones(batch_size), delta_2) / batch_size # shape: (出力層の次元数,)

    # パラメータの更新
    W1 -= eps * dW1
    b1 -= eps * db1

    W2 -= eps * dW2
    b2 -= eps * db2

    return cost

def valid_xor(x, t):
    global W1, b1, W2, b2

    # 順伝播
    u1 = np.matmul(x, W1) + b1
    h1 = relu(u1)

    # 逆伝播
    u2 = np.matmul(h1, W2) + b2
    y = sigmoid(u2)

    # 誤差の計算
    cost = (- t * np_log(y) - (1 - t) * np_log(1 - y)).mean()

    return cost, y



for epoch in range(3000):
    # オンライン学習
    for x, t in zip(x_train_xor, t_train_xor):
        cost = train_xor(x[None, :], t[None, :], eps=0.05)

cost, y_pred = valid_xor(x_valid_xor, t_valid_xor)
print(y_pred)
