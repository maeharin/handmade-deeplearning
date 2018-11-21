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
