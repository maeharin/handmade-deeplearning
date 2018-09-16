package main

import (
	"fmt"
	"math/rand"
	"math"
)

type NeuralNet struct {
	inputNum     int
	hiddenNum    int
	outputNum    int
	learningRate float64
	wIH          Weight // weight between input nodes and hidden nodes
	wHO          Weight // weight between hidden nodes and output nodes
	activateFunc func(input float64) float64
}

type Weight [][]float64

func CreateNeuralNet(inputNum int, hiddenNum int, outputNum int, learningRate float64) *NeuralNet {
	nn := &NeuralNet{
		inputNum:     inputNum, // 入力層の次元（特徴量の変数の数）
		hiddenNum:    hiddenNum, // 隠れ層の次元（任意？）
		outputNum:    outputNum, // 出力層の次元（正解カテゴリをone-hotベクトルで表したもの）
		learningRate: learningRate, // 学習率
	}

	makeWeight := func(dim1, dim2 int) Weight{
		weight := make([][]float64, dim1)
		for i := 0; i < dim1; i++ {
			weight[i] = make([]float64, dim2)
		}
		for i := 0; i < dim1; i++ {
			for j := 0; j < dim2; j++ {
				weight[i][j] = rand.Float64()
			}
		}
		return weight
	}

	// input > hidden間の重み
	nn.wIH = makeWeight(hiddenNum, inputNum)
	// hidden > output間の重み
	nn.wHO = makeWeight(outputNum, hiddenNum)

	nn.activateFunc = sigmoid

	return nn
}

// シグモイド関数
// 前の層の出力と重みを掛け合わせた行列をうけて、0 < 1の出力におきかえる活性化関数
func sigmoid(input float64) float64 {
	return 1.0 / (1.0 + math.Exp(-input))
}

// 一つのレコードごとに訓練する
// data: x
// label: y
func (nn *NeuralNet) Train(inputs []float64, targets []float64) {
	fmt.Printf("-------------------------\n")
	fmt.Printf("train start...\n")
	fmt.Printf("-------------------------\n")
	fmt.Printf("wIH: %v\n", nn.wIH)
	fmt.Printf("wHO: %v\n", nn.wHO)
	fmt.Printf("inputs: %v\n", inputs)
	fmt.Printf("targets: %v\n", targets)
	// dataをニューラルネットに通して、outputを得る
	// input > hidden > output

	// 隠れ層への入力
	hiddenInputs := Dot(inputs, nn.wIH)
	fmt.Printf("hiddenInputs: %v\n", hiddenInputs)

	// 隠れ層の出力
	hiddenOutpus := make([]float64, nn.hiddenNum)
	for i, hi := range hiddenInputs {
		hiddenOutpus[i] = nn.activateFunc(hi)
	}
	fmt.Printf("hiddenOutputs: %v\n", hiddenOutpus)

	// 出力層への入力
	outputInputs := Dot(hiddenOutpus, nn.wHO)

	// 最終出力
	finalOutputs := make([]float64, nn.outputNum)
	for i, oi := range outputInputs {
		finalOutputs[i] = nn.activateFunc(oi)
	}
	fmt.Printf("finalOutputs: %v\n", finalOutputs)

	// 最終出力とtargetの誤差をとる
	errors := make([]float64, nn.outputNum)
	for i := 0; i < nn.outputNum; i++ {
		// 誤差を2乗
		e := targets[i] - finalOutputs[i]
		errors[i] = e * e
	}
	fmt.Printf("errors: %v\n", errors)

	// 誤差を逆伝播させてニューラルネットの重みを更新する

	// wHOを更新
	// wHO => Eを最小化する方向に向かってwHOを更新する
	// つまりdE/dwHO（wHOと最終誤差の変化率、傾き）を求めて、それを0に学習率分近づける
	// この重み更新分をdWとすると
	// dW := nn.learningRate * (errors * finalOutputs * (1-finalOutputs)) dot hiddenOutpus.T)



	// 入力層と隠れ層の誤差は、最終層の誤差をリンクの重みで分配

	// 入力層と隠れ層間の重みを更新
}

func (nn *NeuralNet) Predict() {
	fmt.Println("predict...")
}

// ベクトルと行列の積をとる
func Dot(inputs []float64, w Weight) []float64 {
	num := len(w)
	res := make([]float64, num)
	for n := 0; n < num; n++ {
		// 入力層の要素分、重みをかけて足し込む
		for i, input := range inputs {
			res[n] += input * w[n][i]
		}
	}
	return res
}

func main() {
	data := [][]float64 {
		{7.0, 8.0},
		{0.1, 0.2},
		{0.1, 0.2},
	}

	label := [][]float64 {
		{0, 1, 0},
		{1, 1, 0},
		{1, 1, 1},
	}

	// ニューラルネット構築
	// inputNumはdataの特徴量の変数の数（上記の例だと2個）
	nn := CreateNeuralNet(2, 3, 3, 0.3)

	for i, d := range data {
		l := label[i]
		nn.Train(d, l)
	}
}

