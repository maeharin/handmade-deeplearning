import numpy as np
import sys

np.random.seed(34)


# np.log(x)のxに0がくると、divide by zero encountered in logというエラーになる
# xの最小値を0に極めて近い値（1e-10とか）にする
# minをnp.clip(配列,min,max)をつかうと、min以下はminに。max以上はmaxにおさめてくれるのでこれを使う
def np_log(x):
    return np.log(np.clip(x, 1e-10, x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def cost_softmax(t, y):
    return (- t * np_log(y)).sum().mean()
    # 松尾研で以下のようになっているのは、ミニバッチを想定しているから？
    # return (- t * np_log(y)).sum(axis=1).mean()


def cost_sigmoid(t, y):
    return (- t * np_log(y) - (1 - t) * np_log(1 - y)).mean()


if __name__ == "__main__":
    # np.random.uniformはlow-highの範囲で一様分布の乱数を生成
    # xは5 x 3（3つの要素をもつ横Vectorが5つ）
    # 中間層のUnit数を4とするとすると, Wは3x4
    # xとWの積は、4要素の横Vectorになる
    W1 = np.random.uniform(low=-0.08, high=0.08, size=(3, 4))
    # 6クラスに分類するとする
    # 4要素のVectorに4x6の重みをかける
    W2 = np.random.uniform(low=-0.08, high=0.08, size=(4, 6))

    x_train = np.arange(0, 15).reshape(5, 3)
    # 5 x 1 => 5 x 6（one-hot)
    t_train = np.array([5,3,0,2,4])
    t_train = np.eye(6)[t_train]

    for x, t in zip(x_train, t_train):
        print("---")
        h = np.matmul(x, W1)
        h = sigmoid(h)
        h = np.matmul(h, W2)
        y = softmax(h)

        # 6要素のVector（one-hotに対応）
        print(y)
        print(t)

        # 誤差（クロスエントロピーloss、最終層がsoftmaxの場合のパターン？
        cost = cost_softmax(t, y)
        print(cost)

        # todo: backprop

        print("---")




