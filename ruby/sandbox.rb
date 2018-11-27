require './util'

require 'pp'

RED="\033[0;31m"
GREEN="\033[0;32m"

def assert(a,b)
  if a != b
    print "#{RED} error. expect:#{b} actual:#{a}\n"
  else
    print "#{GREEN} ok. expect:#{b} actual:#{a}\n"
  end
end

x = [1, 2]
w = [
  [10, 20, 30],
  [100, 200, 300]
]
assert(matmul(x, w), [210, 420, 630])

a = [1,2]
b = [10,20,30]
assert(vv(a,b), [[10, 20, 30], [20, 40, 60]])

pp a.transpose

