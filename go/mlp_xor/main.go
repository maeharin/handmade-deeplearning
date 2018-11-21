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
	fmt.Println(w1)
	fmt.Println(b1)
	fmt.Println(w2)
	fmt.Println(b2)

	for i := 0; i < len(xTrain); i++ {
		x := xTrain[i]
		t := tTrain[i]
		fmt.Println(x, t)
	}
}

