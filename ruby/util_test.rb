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

assert(relu(2), 2)
assert(relu(0), 0)
assert(relu(-1), 0)

assert(deriv_relu(2), 1)
assert(deriv_relu(0), 0)
assert(deriv_relu(-1), 0)

assert(sigmoid(0), 0.5)



x = [1, 2]
w = [
  [10, 20, 30],
  [100, 200, 300]
]
assert(matmul(x, w), [210, 420, 630])
