package main

import (
	"math"
	"math/rand"
)

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func DerivSigmoid(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

func Relu(x float64) float64 {
	if x > 0 {
		return x
	} else {
		return 0
	}
}

func DerivRelu(x float64) float64 {
	if x > 0 {
		return 1
	} else {
		return 0
	}
}

func randU(min float64, max float64) float64 {
	return rand.Float64() * (max - min) + min
}

func CreateWeight(inDim int, outDim int) [][]float64 {
	w := make([][]float64, inDim)
	for i := 0; i < inDim; i++ {
		w[i] = make([]float64, outDim)
		for j := 0; j < outDim; j++ {
			w[i][j] = randU(-0.08, 0.08)
		}
	}
	return w
}

func CreateBias(outDim int) []float64 {
	b := make([]float64, outDim)
	for i := 0; i < outDim; i++ {
		b[i] = 0
	}
	return b
}

// [x0, x1]
// *
// [
//   [w00, w01, w02],
//   [w10, w11, w12]
// ]
// =>
// [u0, u1, u2]
//
func Matmul(x []float64, w [][]float64) []float64 {
	inDim := len(x)
	outDim := len(w[0])
	res := make([]float64, outDim)
	for i := 0; i < outDim; i++ {
		res[i] = 0
		for j := 0; j < inDim; j++ {
			res[i] += x[j] * w[j][i]
		}
	}
	return res
}

// {1,2,3},
// {4,5,6},
// =>
// {1, 4},
// {2, 5},
// {3, 6},
func Transpose(w [][]float64) [][]float64 {
	dim1 := len(w)
	dim2 := len(w[0])
	t := make([][]float64, dim2)
	for i := 0; i < dim2; i++ {
		t[i] = make([]float64, dim1)
		for j := 0; j < dim1; j++ {
			t[i][j] = w[j][i]
		}
	}
	return t
}

// a = [1,2]
// b = [10,20,30]
// assert(vv(a,b), [[10, 20, 30], [20, 40, 60]])
func Vv(v1 []float64, v2[]float64) [][]float64 {
	w := make([][]float64, len(v1))
	for i := 0; i < len(v1); i++ {
		w[i] = make([]float64, len(v2))
		for j := 0; j < len(v2); j++ {
			w[i][j] = v1[i] * v2[j]
		}
	}
	return w
}

func UpdateWeight(w [][]float64, dw[][]float64, lr float64) [][]float64 {
	rowDim := len(w)
	colDim := len(w[0])
	ww := make([][]float64, rowDim)
	for i := 0; i < rowDim; i++ {
		ww[i] = make([]float64, colDim)
		for j := 0; j < colDim; j++ {
			ww[i][j] = w[i][j] - (lr * dw[i][j])
		}
	}
	return ww
}

func UpdateBias(b []float64, db []float64, lr float64) []float64 {
	bb := make([]float64, len(b))
	for i := 0; i < len(b); i++ {
		bb[i] = b[i] - (lr * db[i])
	}
	return bb
}
