#
# XOR classifier
#

require 'pp'
require './util'

# XOR
X = [
  [0,1],
  [1,0],
  [0,0],
  [1,1]
]
T = [
  [1],
  [1],
  [0],
  [0]
]

srand(34)
w1 = create_weight(2, 8)
b1 = create_bias(8)
w2 = create_weight(8, 1)
b2 = create_bias(1)

# 注意: trainのところと同じになるように
def forward(x, w1, w2, b1, b2)
  u1 = matmul(x, w1)
  u1 = u1.zip(b1).map { |u,b| u + b }
  h1 = u1.map {|v| relu(v) }
  u2 = matmul(h1, w2)
  u2 = u2.zip(b2).map { |u,b| u + b }
  y = u2.map {|v| sigmoid(v)}
  [u1, h1, u2, y]
end

(0...3000).each do |epoch|
  X.zip(T).each do |x, t|
    # forward
    u1, h1, u2, y = forward(x, w1, w2, b1, b2)

    # backprop
    # y,tは配列（要素数1）
    delta2 = y.zip(t).map { |yy, tt| yy - tt }
    a = matmul(delta2, w2.transpose)
    delta1 = a.zip(u1).map { |aa, uu1| aa * deriv_relu(uu1) }

    # 勾配
    dw1 = vv(x, delta1)
    dw2 = vv(h1, delta2)
    db1 = delta1
    db2 = delta2

    # 重み更新　
    lr = 0.05
    w1 = mat_minus(w1, dw1, lr)
    w2 = mat_minus(w2, dw2, lr)
    b1 = b1.zip(db1).map { |b, d| b - lr * d }
    b2 = b2.zip(db2).map { |b, d| b - lr * d }
  end
end



X.each do |x|
  _, _, _, y = forward(x, w1, w2, b1, b2)
  y = y[0]
  p "#{x}: " + sprintf("%.5f", y)
end

#
# gradient checking
#

def calc_cost(t, y)
  # maximize: y^t * (1-y)^(1-t)
  # =>
  # minimize: log(y^t * (1-y)^(1-t))
  # = log(y^t) + log(1-y)^(1-t)
  # = t * log(y) + (1-t) * log(1-y)
  - (t * Math.log(y) + (1-t) * Math.log(1-y))
end

# weights and biases for gradient che3cking
w1 = create_weight(2, 8)
b1 = create_bias(8)
w2 = create_weight(8, 1)
b2 = create_bias(1)

# data for gradient checking
x = X[0]
t = T[0]

#
# compute graditents by back propagation
#
u1, h1, u2, y = forward(x, w1, w2, b1, b2)
delta2 = y.zip(t).map {|yy,tt| yy - tt}
a = matmul(delta2, w2.transpose)
delta1 = a.zip(u1).map {|aa, uu1| aa * deriv_relu(uu1)}
dw2 = vv(h1, delta2)
dw1 = vv(x, delta1)
db2 = delta2
db1 = delta1

#
# compute graditents by numerical grad
#

h = 1e-5

# w2
n_dw2 = w2.map.with_index do |row, i|
  row.map.with_index do |v, j|
    w2[i][j] = v + h
    _, _, _, y = forward(x, w1, w2, b1, b2)
    cost_plus = calc_cost(t[0], y[0])
    w2[i][j] = v - h
    _, _, _, y = forward(x, w1, w2, b1, b2)
    cost_minus = calc_cost(t[0], y[0])
    grad = (cost_plus - cost_minus) / (2 * h)
    w2[i][j] = v # 元に戻す
    grad
  end
end

# w1
n_dw1 = w1.map.with_index do |row, i|
  row.map.with_index do |v, j|
    w1[i][j] = v + h
    _, _, _, y = forward(x, w1, w2, b1, b2)
    cost_plus = calc_cost(t[0], y[0])
    w1[i][j] = v - h
    _, _, _, y = forward(x, w1, w2, b1, b2)
    cost_minus = calc_cost(t[0], y[0])
    grad = (cost_plus - cost_minus) / (2 * h)
    w1[i][j] = v
    grad
  end
end

# b2
# b1
#
p "dw2: #{dw2}"
p "n_dw2: #{n_dw2}"

p "dw1: #{dw1}"
p "n_dw1: #{n_dw1}"
