package main

import (
	"testing"
)

func TestSigmoid(t *testing.T) {
	actual := Sigmoid(0)
	expected := 0.5
	if actual != expected {
		t.Errorf("expected: %v but %v", expected, actual)
	}
}

func TestDerivSigmoid(t *testing.T) {
	actual := DerivSigmoid(0)
	expected := 0.25
	if actual != expected {
		t.Errorf("expected: %v but %v", expected, actual)
	}
}

func TestRelu(t *testing.T) {
	data := [][]float64 {
		{-1, 0},
		{0, 0},
		{1, 1},
		{2, 2},
	}
	for _, d := range(data) {
		actual := Relu(d[0])
		expected := d[1]
		if actual != expected {
			t.Errorf("expected: %v but %v", expected, actual)
		}
	}
}

func TestDerivRelu(t *testing.T) {
	data := [][]float64 {
		{-1, 0},
		{0, 0},
		{1, 1},
		{2, 1},
	}
	for _, d := range(data) {
		actual := DerivRelu(d[0])
		expected := d[1]
		if actual != expected {
			t.Errorf("expected: %v but %v", expected, actual)
		}
	}
}