import numpy as np
from util import *
import sys as sys

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
w1_in_dim = 2
w1_out_dim = 8
w1 = np.random.uniform(-0.08, 0.08, (w1_in_dim, w1_out_dim)).astype('float32')
b1 = np.zeros(w1_out_dim).astype('float32')

w2_in_dim = 8
w2_out_dim = 1
w2 = np.random.uniform(-0.08, 0.08, (w2_in_dim, w2_out_dim)).astype('float32')
b2 = np.zeros(w2_out_dim).astype('float32')


# train
for epoch in range(0, 3000):
    for x, t in zip(X, T):
        # forward
        u1 = np.matmul(x, w1) + b1
        h1 = relu(u1)
        u2 = np.matmul(h1, w2) + b2
        y = sigmoid(u2)

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

def pred(x):
    u1 = np.matmul(x, w1) + b1
    h1 = relu(u1)
    u2 = np.matmul(h1, w2) + b2
    y = sigmoid(u2)
    return y

for x in X:
    res = pred(x)
    print(x,res)
