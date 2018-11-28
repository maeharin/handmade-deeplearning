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
    delta1 = begin
      a = matmul(delta2, w2.transpose)
      b = u1.map{ |v| deriv_relu(v) }
      a.zip(b).map { |c,d| c * d }
    end

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


# gradient checking

# backpropagation

w1 = create_weight(2, 8)