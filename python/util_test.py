import unittest
import numpy as np
from util import *

np.random.seed(34)

class TestHoge(unittest.TestCase):
    def test_relu(self):
        actual = relu(np.array([2, 0, -1]))
        expect = np.array([2, 0, 0])
        np.testing.assert_allclose(actual, expect)

    def test_deriv_relu(self):
        actual = deriv_relu(np.array([2, 0, -1]))
        expect = np.array([1, 0, 0])
        np.testing.assert_allclose(actual, expect)

    def test_sigmoid(self):
        self.assertEqual(sigmoid(0), 0.5)
        self.assertEqual(sigmoid(1000), 1.0)
        self.assertEqual(sigmoid(-1000), 0.0)

    @unittest.skip
    def test_softmax(self):
        pass

    def test_shuffle(self):
        a = np.array([
            [1,2,3],
            [10,20,30],
            [100,20,30],
        ])
        b = np.array([1,2,3])
        c, d = shuffle(a, b)

        np.testing.assert_allclose(c, np.array([
            [100,20,30],
            [1,2,3],
            [10,20,30],
        ]))
        np.testing.assert_allclose(d, np.array([3,1,2]))



if __name__ == '__main__':
    unittest.main()
