#
# 2layer multi parceptron layer
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
  1,
  1,
  0,
  0
]

# 重み初期化
# todo うーん。。。ランダムのシード値によって学習できるかどうかが変わってくるなぁ。。。。
RSEED = 34
r = Random.new(RSEED)

w1_in_dim = 2
w1_out_dim = 8
p "w1 shape: (#{w1_in_dim},#{w1_out_dim})"
w1 = []
(0..w1_in_dim - 1).each do |i|
  (0..w1_out_dim - 1).each do |j|
    w1[i] = [] if w1[i].nil?
    w1[i][j] = r.rand(-0.08..0.08)
  end
end

# b1: (,8)
b1 = []
(0..w1_out_dim - 1).each do |i|
  b1[i] = 0
end

# w2: (8,1)
w2_in_dim = 8
w2_out_dim = 1
p "w2 shape: (#{w2_in_dim},#{w2_out_dim})"
w2 = []
(0..w2_in_dim - 1).each do |i|
  (0..w2_out_dim - 1).each do |j|
    w2[i] = [] if w2[i].nil?
    w2[i][j] = r.rand(-0.08..0.08)
  end
end

# b2: (,1)
b2 = []
(0..w2_out_dim - 1).each do |i|
  b2[i] = 0
end

p "initial w, b"
p "w1: #{w1}"
p "b1: #{b1}"
p "w2: #{w2}"
p "b2: #{b2}"

(0..3000).each do |epoch|
  X.zip(T).each do |x, t|
    # forward
    u1 = matmul(x, w1)
    u1 = u1.zip(b1).map { |u,b| u + b }
    h1 = u1.map {|v| relu(v) }
    u2 = matmul(h1, w2)
    u2 = u2.zip(b2).map { |u,b| u + b }
    y = u2.map {|v| sigmoid(v)}

    # backward
    # yは配列（要素数1）
    delta2 = [y[0] - t]
    delta1 = begin
      a = matmul(delta2, w2.transpose)
      b = u1.map{ |v| deriv_relu(v) }
      a.zip(b).map { |c,d| c * d }
    end

    # 勾配
    dw1 = vv(x, delta1)
    dw2 = vv(h1, delta2)
    # うーん。。。。ここはどうなるのか
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

# 注意: trainのところと同じになるように
def pred(x, w1, w2, b1, b2)
    u1 = matmul(x, w1)
    u1 = u1.zip(b1).map { |u,b| u + b }
    h1 = u1.map {|v| relu(v) }
    u2 = matmul(h1, w2)
    u2 = u2.zip(b2).map { |u,b| u + b }
    y = u2.map {|v| sigmoid(v)}
    y
end


X.each do |x|
  y = pred(x, w1, w2, b1, b2)[0]
  p "#{x}: " + sprintf("%.5f", y)
end
