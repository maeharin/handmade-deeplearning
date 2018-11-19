require 'pp'
require './util'

X = [
  [1,1],
  [1,0],
  [0,1],
  [0,0],
]

# AND
T = [
  1,
  0,
  0,
  0,
]

# 重み初期化
r = Random.new(1234)

# w1: (2,3)
w1 = []
(0..1).each do |i|
  (0..2).each do |j|
    w1[i] = [] if w1[i].nil?
    w1[i][j] = r.rand(-0.08..0.08)
  end
end

# w2: (3,1)
w2 = []
(0..2).each do |i|
  (0..0).each do |j|
    w2[i] = [] if w2[i].nil?
    w2[i][j] = r.rand(-0.08..0.08)
  end
end

(0..3000).each do |epoch|
  X.zip(T).each do |x, t|
    # forward
    u1 = matmul(x, w1)
    h1 = u1.map {|v| sigmoid(v) }
    u2 = matmul(h1, w2)
    y = u2.map {|v| sigmoid(v)}

    # backward
    # yが配列（要素数1）なのでtも配列に
    delta2 = y - [t]
    delta1 = begin
      a = matmul(delta2, w2.transpose)
      b = u1.map{ |v| deriv_sigmoid(v) }
      a.zip(b).map { |c,d| c * d }
    end

    # 勾配
    dw1 = vv(x, delta1)
    dw2 = vv(h1, delta2)

    # 重み更新　
    lr = 0.01
    w1 = mat_minus(w1, dw1, lr)
    w2 = mat_minus(w2, dw2, lr)
  end
end

def pred(x, w1, w2)
  u1 = matmul(x, w1)
  h1 = u1.map {|v| sigmoid(v) }
  u2 = matmul(h1, w2)
  u2.map {|v| sigmoid(v)}
end

X.each do |x|
  y = pred(x, w1, w2)
  p "#{x}: #{y}"
end
