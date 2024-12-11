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

func NewLayerDense(inputs, neurons int) *LayerDense {
	// Min/max for rand weights
	const (
		min = -0.2
		max = 0.2
	)

	rdata := make([]float64, inputs*neurons)
	for i := 0; i < inputs*neurons; i++ {
		rdata[i] = min + rand.Float64()*(max-min)
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

	for i := 0; i < layer.output.RawMatrix().Rows; i++ {
		for j := 0; j < layer.output.RawMatrix().Cols; j++ {
			layer.output.Set(i, j, layer.output.At(i, j)+layer.biases.At(0, j))
		}
	}
}

func main() {
	// The book uses spiral data here but sticking
	// with the simple example as per youtube
	X := mat.NewDense(3, 4, []float64{
		1, 2, 3, 2.5,
		2.0, 5.0, -1.0, 2.0,
		-1.5, 2.7, 3.3, -0.8,
	})

	layer1 := NewLayerDense(4, 5)
	layer1.Forward(X)
	fmt.Println(mat.Formatted(layer1.output))

	layer2 := NewLayerDense(5, 2)
	layer2.Forward(layer1.output)
	fmt.Println(mat.Formatted(layer2.output))
}
