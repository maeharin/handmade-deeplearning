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

func TestTranspose(t *testing.T) {
	w := [][]float64 {
		{1,2,3},
		{4,5,6},
	}
	actual := Transpose(w)
	expected := [][]float64 {
		{1, 4},
		{2, 5},
		{3, 6},
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected: %v but %v", expected, actual)
	}
}

func TestVv(t *testing.T) {
	a := []float64{1, 2}
	b := []float64{10, 20, 30}
	actual := Vv(a, b)
	expected := [][]float64 {
		{10, 20, 30},
		{20, 40, 60},
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected: %v but %v", expected, actual)
	}
}

// a = [
//  [1,2,3],
//  [4,5,6]
//]
//b = [
//  [1,2,3],
//  [4,5,6]
//]
//lr = 0.1
//# 1 - 1 * 0.1 = 0.9
//# 2 - 2 * 0.1 = 1.8
//# 3 - 3 * 0.1 = 2.7
//# 4 - 4 * 0.1 = 3.6
//# 5 - 5 * 0.1 = 4.5
//# 6 - 6 * 0.1 = 5.4
//assert(mat_minus(a, b, lr), [
//  [0.9, 1.8, 2.7],
//  [3.6, 4.5, 5.4]
//])
func TestUpdateWeight(t *testing.T) {
	w := [][]float64{
		{1,2,3},
		{4,5,6},
	}
	dw := [][]float64{
		{1,2,3},
		{4,5,6},
	}
	lr := 0.1
	actual := UpdateWeight(w, dw, lr)
	expected := [][]float64 {
		{0.9, 1.8, 2.7},
		{3.6, 4.5, 5.4},
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected: %v but %v", expected, actual)
	}
}

func TestUpdateBias(t *testing.T) {
	b := []float64{1,2,3}
	db := []float64{1,2,3}
	lr := 0.1
	actual := UpdateBias(b, db, lr)
	expected := []float64{0.9, 1.8, 2.7}
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected: %v but %v", expected, actual)
	}
}