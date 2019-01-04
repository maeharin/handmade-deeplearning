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
# lecun_uniform
# http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
w1 = np.random.uniform(-np.sqrt(1.0/2), np.sqrt(1.0/2), (2, 8)).astype('float32')
b1 = np.zeros(8).astype('float32')
w2 = np.random.uniform(-np.sqrt(1.0/8), np.sqrt(1.0/8), (8, 1)).astype('float32')
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
        w2 -= lr * dW2
        w1 -= lr * dW1
        b2 -= lr * db2
        b1 -= lr * db1


# pred
for x in X:
    _, _, _, y = forward(x, w1, b1, w2, b2)
    print(x,y)


#
# gradient checking
#

# np.logでdivide by zero encountered in logエラーが出るのを回避するためのヘルパ
# aの最小値を決めて分母が0になるのを避ける
def np_log(a):
    return np.log(np.clip(a, 1e-10, a))

# 2値分類なので、尤度関数 = y(t) * (1-y)(1-t) yは確率。t={0,1}
# この尤度関数を最大化したい => 積は面倒なのでlogをとって和の関数にする。符号をマイナスに
# cost = log(y(t) * (1-y)(1-t))
# cost = log(y(t)) + log((1-y)(1-t))
# cost = {t*log(y) + (1-t)*log(1-y)}
# cost = t*log(y) + (1-t)*log(1-y)}
# cost = - { t*log(y) + (1-t)*log(1-y)} }
def calc_cost(t, y):
    return - (t * np_log(y) + (1 - t) * np_log(1 - y))

w1 = np.random.uniform(-0.08, 0.08, (2, 8)).astype('float32')
b1 = np.zeros(8).astype('float32')
w2 = np.random.uniform(-0.08, 0.08, (8, 1)).astype('float32')
b2 = np.zeros(1).astype('float32')
for x, t in zip(X, T):
    #
    # compute graditents by back propagation
    #

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

    #
    # compute graditents by numerical grad
    #

    # forward
    _, _, _, y = forward(x, w1, b1, w2, b2)

    # calc numerical grad
    # コスト関数をw1,b1,w2,b2でそれぞれ数値微分（+h,-hでそれぞれ計算する）して勾配を求める
    h = 1e-5 # 0.00001

    # w1
    # w1（行列）の値を一個ずつ+h,-hで微分していく
    n_dW1 = np.zeros_like(w1)
    for i, w1_row in enumerate(w1):
        for j, _ in enumerate(w1_row):
            # i,jの位置の値だけ+hした重みで予測をしてcost_hを取得し、上述のcostとの差をとって微分
            tmp = w1[i][j]
            # plus
            w1[i][j] = tmp + h
            _, _, _, y = forward(x, w1, b1, w2, b2)
            cost_plus = calc_cost(t, y)
            # minus
            w1[i][j] = tmp - h
            _, _, _, y = forward(x, w1, b1, w2, b2)
            cost_minus = calc_cost(t, y)
            # grad
            n_dW1[i][j] = (cost_plus - cost_minus) / (2 * h)
            w1[i][j] = tmp  # 元に戻す

    # db1
    n_db1 = np.zeros_like(b1)
    for i, _ in enumerate(b1):
        tmp = b1[i]
        # plus
        b1[i] = tmp + h
        _, _, _, y = forward(x, w1, b1, w2, b2)
        cost_plus = calc_cost(t, y)
        # minus
        b1[i] = tmp - h
        _, _, _, y = forward(x, w1, b1, w2, b2)
        cost_minus = calc_cost(t, y)
        # grad
        n_db1[i] = (cost_plus - cost_minus) / (2 * h)
        b1[i] = tmp  # 元に戻す

    # dw2
    n_dW2 = np.zeros_like(w2)
    for i, w2_row in enumerate(w2):
        for j, _ in enumerate(w2_row):
            tmp = w2[i][j]
            w2[i][j] = tmp + h
            _, _, _, y = forward(x, w1, b1, w2, b2)
            cost_plus = calc_cost(t, y)
            w2[i][j] = tmp - h
            _, _, _, y = forward(x, w1, b1, w2, b2)
            cost_minus = calc_cost(t, y)
            n_dW2[i][j] = (cost_plus - cost_minus) / (2 * h)
            w2[i][j] = tmp

    # db2
    n_db2 = np.zeros_like(b2)
    for i, _ in enumerate(b2):
        tmp = b2[i]
        b2[i] = tmp + h
        _, _, _, y = forward(x, w1, b1, w2, b2)
        cost_plus = calc_cost(t, y)
        b2[i] = tmp - h
        cost_minus = calc_cost(t, y)
        n_db2[i] = (cost_plus - cost_minus) / (2 * h)
        b2[i] = tmp

    #print("dW1:", dW1)
    #print("n_dW1:", n_dW1)
    #print("db1:", db1)
    #print("n_db1:", n_db1)
    #print("dW2:", dW2)
    #print("n_dW2:", n_dW2)
    #print("db2:", db2)
    #print("n_db2:", n_db2)
    diff = np.average(np.abs(dW1 - n_dW1))
    print("diff: %.10f" % diff)




