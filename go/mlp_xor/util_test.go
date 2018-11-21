package main

import (
	"reflect"
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

func TestMatmul(t *testing.T) {
	x := []float64{1,2}
	w := [][]float64{
		{10, 20, 30},
		{100,200,300},
	}
	actual := Matmul(x, w)
	expected := []float64 {210, 420, 630}
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected: %v but %v", expected, actual)
	}
}