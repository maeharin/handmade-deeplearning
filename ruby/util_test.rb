require './util'

RED="\033[0;31m"
GREEN="\033[0;32m"

def assert(a,b)
  if a != b
    print "#{RED} error. expect:#{b} actual:#{a}\n"
  else
    print "#{GREEN} ok. expect:#{b} actual:#{a}\n"
  end
end

p "test start..."

p "test relu"
assert(relu(2), 2)
assert(relu(0), 0)
assert(relu(-1), 0)

p "test deriv_relu"
assert(deriv_relu(2), 1)
assert(deriv_relu(0), 0)
assert(deriv_relu(-1), 0)

p "test sigmoidj"
assert(sigmoid(1000), 1.0)
assert(sigmoid(0), 0.5)
assert(sigmoid(-1000), 0.0)

p "test matmul"
x = [1, 2]
w = [
  [10, 20, 30],
  [100, 200, 300]
]
assert(matmul(x, w), [210, 420, 630])

p "test vv"
a = [1,2]
b = [10,20,30]
assert(vv(a,b), [[10, 20, 30], [20, 40, 60]])

p "test mat_minus"
a = [
  [1,2,3],
  [4,5,6]
]
b = [
  [1,2,3],
  [4,5,6]
]
lr = 0.1
# 1 - 1 * 0.1 = 0.9
# 2 - 2 * 0.1 = 1.8
# 3 - 3 * 0.1 = 2.7
# 4 - 4 * 0.1 = 3.6
# 5 - 5 * 0.1 = 4.5
# 6 - 6 * 0.1 = 5.4
assert(mat_minus(a, b, lr), [
  [0.9, 1.8, 2.7],
  [3.6, 4.5, 5.4]
])

p "test end"
