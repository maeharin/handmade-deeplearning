package main

import (
	"testing"
)

func TestDot(t *testing.T) {
	i := []float64{7, 8}
	w := Weight{
		{1, 2},
		{3, 4},
		{5, 6},
	}
	tests := []float64{23,53,83}
	o := Dot(i, w)
	for i, tt := range tests {
		if o[i] != tt {
			t.Fatalf("error! expected: %v, but: %v", tt, o[i])
		}
	}
}

