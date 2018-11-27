import numpy as np
from util import *

# XOR
X = np.array([
  [0,1],
  [1,0],
  [0,0],
  [1,1]
])

T = np.array([
  [1],
  [1],
  [0],
  [0]
])

np.random.seed(34)

# 重み初期化
w1 = np.random.uniform(-0.08, 0.08, (2, 8)).astype('float32')
b1 = np.zeros(8).astype('float32')
w2 = np.random.uniform(-0.08, 0.08, (8, 1)).astype('float32')
b2 = np.zeros(1).astype('float32')


def forward(x, w1, b1, w2, b2):
    u1 = np.matmul(x, w1) + b1
    h1 = relu(u1)
    u2 = np.matmul(h1, w2) + b2
    y = sigmoid(u2)
    return u1, h1, u2, y


# train
for epoch in range(0, 3000):
    for x, t in zip(X, T):
        # forward
        u1, h1, u2, y = forward(x, w1, b1, w2, b2)

        # backprop
        delta2 = y - t
        delta1 = np.matmul(delta2, w2.T) * deriv_relu(u1)

        # 勾配
        dW2 = np.matmul(h1[None, :].T, delta2[None, :])
        dW1 = np.matmul(x[None, :].T, delta1[None, :])
        db2 = delta2
        db1 = delta1

        # 重み更新　
        lr = 0.05
        w2 = w2 - lr * dW2
        w1 = w1 - lr * dW1
        b2 = b2 - lr * db2
        b1 = b1 - lr * db1


# pred
for x in X:
    _, _, _, y = forward(x, w1, b1, w2, b2)
    print(x,y)


#
# gradient checking
#

# backprop
w1 = np.random.uniform(-0.08, 0.08, (2, 8)).astype('float32')
b1 = np.zeros(8).astype('float32')
w2 = np.random.uniform(-0.08, 0.08, (8, 1)).astype('float32')
b2 = np.zeros(1).astype('float32')
for x, t in zip(X, T):
    # forward
    u1, h1, u2, y = forward(x, w1, b1, w2, b2)

    # backprop
    delta2 = y - t
    delta1 = np.matmul(delta2, w2.T) * deriv_relu(u1)

    # 勾配
    dW2 = np.matmul(h1[None, :].T, delta2[None, :])
    dW1 = np.matmul(x[None, :].T, delta1[None, :])
    db2 = delta2
    db1 = delta1

    # 重み更新　
    lr = 0.05
    w2 = w2 - lr * dW2
    w1 = w1 - lr * dW1
    b2 = b2 - lr * db2
    b1 = b1 - lr * db1

# np.logでdivide by zero encountered in logエラーが出るのを回避するためのヘルパ
# aの最小値を決めて分母が0になるのを避ける
def np_log(a):
    return np.log(np.clip(a, 0.0000001, a))

n_w1 = np.random.uniform(-0.08, 0.08, (2, 8)).astype('float32')
n_b1 = np.zeros(8).astype('float32')
n_w2 = np.random.uniform(-0.08, 0.08, (8, 1)).astype('float32')
n_b2 = np.zeros(1).astype('float32')
for x, t in zip(X, T):
    # numerical grad
    # - コスト関数を定義
    # 2値分類なので、尤度関数 = y(t) * (1-y)(1-t) yは確率。t={0,1}
    # この尤度関数を最大化したい => 積は面倒なのでlogをとって和の関数にする。符号をマイナスに
    # cost = log(y(t) * (1-y)(1-t))
    # cost = log(y(t)) + log((1-y)(1-t))
    # cost = {t*log(y) + (1-t)*log(1-y)}
    # cost = t*log(y) + (1-t)*log(1-y)}
    # cost = - { t*log(y) + (1-t)*log(1-y)} }
    def calc_cost(t, y):
        return - (t * np_log(y) + (1 - t) * np_log(1 - y))

    # forward
    _, _, _, y = forward(x, n_w1, n_b1, n_w2, n_b2)
    cost = calc_cost(t, y)

    # - コスト関数をw1,b1,w2,b2でそれぞれ数値微分（-eps, +epsでそれぞれ計算する）して勾配を求める
    eps = 0.0000001

    # dcost / dw1
    _, _, _, y = forward(x, n_w1 + eps, n_b1, n_w2, n_b2)
    cost_eps = calc_cost(t, y)
    dw1 = (cost + cost_eps) / eps
    # dcost / db1
    _, _, _, y = forward(x, n_w1, n_b1 + eps, n_w2, n_b2)
    cost_eps = calc_cost(t, y)
    db1 = (cost + cost_eps) / eps
    # dcost / dw2
    _, _, _, y = forward(x, n_w1, n_b1, n_w2 + eps, n_b2)
    cost_eps = calc_cost(t, y)
    dw2 = (cost + cost_eps) / eps
    # dcost / db2
    _, _, _, y = forward(x, n_w1, n_b1, n_w2, n_b2 + eps)
    cost_eps = calc_cost(t, y)
    db2 = (cost + cost_eps) / eps

    # - 重みバイアスを更新するk
    lr = 0.05
    n_w1 = n_w1 - lr * dw1
    n_b1 = n_b1 - lr * db1
    n_w2 = n_w2 - lr * dw2
    n_b2 = n_b2 - lr * db2


# backpropで更新した重みバイアスと、数値微分で更新した重みバイアスを比較
print("w1", w1)
print("n_w1", n_w1)
