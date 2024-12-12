package main

import (
	"fmt"
	"math/rand/v2"

	"gonum.org/v1/gonum/mat"
)

type LayerDense struct {
	weights *mat.Dense
	biases  *mat.Dense
	output  *mat.Dense
}

type ActivationReLU struct {
	output *mat.Dense
}

func NewLayerDense(inputs, neurons int) *LayerDense {
	rdata := make([]float64, inputs*neurons)
	for i := 0; i < len(rdata); i++ {
		rdata[i] = rand.NormFloat64() * 0.1
	}

	layer := LayerDense{
		weights: mat.NewDense(inputs, neurons, rdata),
		biases:  mat.NewDense(1, neurons, nil),
	}

	return &layer
}

func (layer *LayerDense) Forward(inputs *mat.Dense) {
	layer.output = mat.NewDense(inputs.RawMatrix().Rows, layer.weights.RawMatrix().Cols, nil)
	layer.output.Mul(inputs, layer.weights)

	r, c := layer.output.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			layer.output.Set(i, j, layer.output.At(i, j)+layer.biases.At(0, j))
		}
	}
}

func (act *ActivationReLU) Forward(inputs *mat.Dense) {
	act.output = mat.NewDense(inputs.RawMatrix().Rows, inputs.RawMatrix().Cols, nil)

	f := func(_, _ int, x float64) float64 { return max(0.0, x) }
	act.output.Apply(f, inputs)
}

func main() {
	X, _ := spiral_data()
	layer1 := NewLayerDense(X.RawMatrix().Cols, 5)
	layer1.Forward(X)

	//fmt.Println(mat.Formatted(layer1.output))

	var activation ActivationReLU
	activation.Forward(layer1.output)
	fmt.Println(mat.Formatted(activation.output))
}
