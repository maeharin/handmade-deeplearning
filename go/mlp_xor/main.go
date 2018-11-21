package main

import (
	"fmt"
	"math/rand"
)

func main() {
	xTrain := [][]float64{
		{0, 1},
		{1, 0},
		{0, 0},
		{1, 1},
	}
	tTrain := [][]float64{
		{1},
		{1},
		{0},
		{0},
	}

	// initialize weights and bias
	rand.Seed(34)
	w1 := CreateWeight(2, 8)
	b1 := CreateBias(8)
	w2 := CreateWeight(8, 1)
	b2 := CreateBias(1)

	EPOCH := 3000
	for i := 0; i < EPOCH; i++ {
		for i := 0; i < len(xTrain); i++ {
			x := xTrain[i]
			t := tTrain[i]

			// forward...
			// (2).(2,8)->(8)
			u1 := Matmul(x, w1)
			for i, u := range(u1) {
				u1[i] = u + b1[i]
			}
			h1 := make([]float64, len(u1))
			for i, u := range(u1) {
				h1[i] = Relu(u)
			}
			// (8).(8,1)->(1)
			u2 := Matmul(h1, w2)
			for i, u := range(u2) {
				u2[i] = u + b2[i]
			}
			y := make([]float64, len(u2))
			for i, u := range(u2) {
				y[i] = Sigmoid(u)
			}

			// backword
			delta2 := make([]float64, len(y))
			for i, yv := range(y) {
				delta2[i] = yv - t[i]
			}

			delta1 := make([]float64, len(u1))
			delta1 = Matmul(delta2, Transpose(w2))
			for i, d := range(delta1) {
				delta1[i] = d * DerivRelu(u1[i])
			}

			// gradient
			dw2 := Vv(h1, delta2)
			dw1 := Vv(x, delta1)
			db2 := delta2
			db1 := delta1

			// update weights
			lr := 0.05
			w1 = UpdateWeight(w1, dw1, lr)
			w2 = UpdateWeight(w2, dw2, lr)
			b1 = UpdateBias(b1, db1, lr)
			b2 = UpdateBias(b2, db2, lr)
		}
	}

	pred := func(x []float64) []float64{
		u1 := Matmul(x, w1)
		for i, u := range(u1) {
			u1[i] = u + b1[i]
		}
		h1 := make([]float64, len(u1))
		for i, u := range(u1) {
			h1[i] = Relu(u)
		}

		// (8).(8,1)->(1)
		u2 := Matmul(h1, w2)
		for i, u := range(u2) {
			u2[i] = u + b2[i]
		}
		y := make([]float64, len(u2))
		for i, u := range(u2) {
			y[i] = Sigmoid(u)
		}
		return y
	}

	for _, x := range(xTrain) {
		y := pred(x)
		fmt.Printf("%v: %v\n", x, y[0])
	}
}

