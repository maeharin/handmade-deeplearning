def relu(x)
  x > 0 ? x : 0
end

def deriv_relu(x)
  x > 0 ? 1 : 0
end

def sigmoid(x)
  1.0 / (1.0 + Math.exp(-x))
end

def deriv_sigmoid(x)
  sigmoid(x) * (1 - sigmoid(x))
end

def create_weight(in_dim, out_dim)
  (0...in_dim).map do |_|
    (0...out_dim).map do |_|
      Random.rand(-0.08..0.08)
    end
  end
end

def create_bias(out_dim)
  (0...out_dim).map do |_|
    Random.rand(-0.08..0.08)
  end
end

# x: vector
# w: matrix
def matmul(x, w)
  in_dim = x.count
  out_dim = w[0].count

  res = []
  (0...out_dim).each do |i|
    sum = 0
    (0...in_dim).each do |j|
      sum += x[j] * w[j][i]
    end
    res << sum
  end
  res
end

# うーんどういう関数にすべきか
# (, 2) . (3, ) => (2,3)
def vv(v1, v2)
  vt = [v1].transpose
  vt.map do |v|
    v2.map do |w|
      v[0] * w
    end
  end
end


# 学習率と勾配で更新した重みを返す
def mat_minus(w, dw, lr)
  w.zip(dw).map do |a,b|
    a.zip(b).map do |c,d|
      c - (lr * d)
    end
  end
end
