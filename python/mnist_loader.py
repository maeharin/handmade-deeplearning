import gzip
import struct
import numpy as np


def load_mnist():
    print('loading mnist...')
    x_train = load_images('../data/train-images-idx3-ubyte.gz')
    x_test = load_images('../data/t10k-images-idx3-ubyte.gz')
    t_train = load_labels('../data/train-labels-idx1-ubyte.gz')
    t_test = load_labels('../data/t10k-labels-idx1-ubyte.gz')
    print('loading mnist done')
    return [x_train, t_train, x_test, t_test]


def load_images(file_name):
    with gzip.open(file_name) as f:
        _, n_images, n_rows, n_cols = struct.unpack('>4i', f.read(4 * 4))
        buf = f.read(n_images * n_rows * n_cols)
        data = np.frombuffer(buf, dtype=np.uint8)
        return data.reshape(n_images, n_rows * n_cols)


def load_labels(file_name):
    with gzip.open(file_name) as f:
        _, n_items = struct.unpack('>2i', f.read(4 * 2))
        buf = f.read(n_items)
        return np.frombuffer(buf, dtype=np.uint8)

