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
	tTrain := []float64{
		1,
		1,
		0,
		0,
	}

	// initialize weights and bias
	rand.Seed(34)
	w1 := CreateWeight(2, 8)
	b1 := CreateBias(8)
	w2 := CreateWeight(8, 1)
	b2 := CreateBias(1)

	for i := 0; i < len(xTrain); i++ {
		x := xTrain[i]
		t := tTrain[i]
		fmt.Println(x)
		fmt.Println(t)

		// forward...
		u1 := Matmul(x, w1)
		for i, u := range(u1) {
			u1[i] = u + b1[i]
		}
		h1 := make([]float64, len(u1))
		for i, u := range(u1) {
			h1[i] = Relu(u)
		}

		u2 := Matmul(h1, w2)
		for i, u := range(u2) {
			u2[i] = u + b2[i]
		}
		y := make([]float64, len(u2))
		for i, u := range(u2) {
			y[i] = Sigmoid(u)
		}

		fmt.Println(y)
		// todo: backword
	}
}

