from util import *

assert(sigmoid(0) == 0.5)

assert(relu(2) == 2)
assert(relu(0) ==  0)
assert(relu(-1) == 0)

x = np.array([2, 0, -1])
print(sigmoid(x))
print(relu(x))
print(deriv_relu(x))

