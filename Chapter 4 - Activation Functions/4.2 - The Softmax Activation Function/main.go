package main

import (
	"fmt"
	"math"
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

type ActivationSoftmax struct {
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

func NewActivationReLU() *ActivationReLU {
	return new(ActivationReLU)
}

func (act *ActivationReLU) Forward(inputs *mat.Dense) {
	act.output = mat.NewDense(inputs.RawMatrix().Rows, inputs.RawMatrix().Cols, nil)

	f := func(_, _ int, x float64) float64 { return max(0.0, x) }
	act.output.Apply(f, inputs)
}

func NewActivationSoftmax() *ActivationSoftmax {
	return new(ActivationSoftmax)
}

func (act *ActivationSoftmax) Forward(inputs *mat.Dense) {
	r, c := inputs.Dims()
	act.output = mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		maximum := inputs.At(i, 0)
		for j := 1; j < c; j++ {
			maximum = max(maximum, inputs.At(i, j))
		}
		sum := 0.0
		for j := 0; j < c; j++ {
			act.output.Set(i, j, math.Exp(inputs.At(i, j)-maximum))
			sum += act.output.At(i, j)
		}
		for j := 0; j < c; j++ {
			act.output.Set(i, j, act.output.At(i, j)/sum)
		}
	}
}

func main() {
	X, _ := spiral_data()

	dense1 := NewLayerDense(X.RawMatrix().Cols, 3)
	activation1 := NewActivationReLU()

	dense2 := NewLayerDense(3, 3)
	activation2 := NewActivationSoftmax()

	dense1.Forward(X)
	activation1.Forward(dense1.output)

	dense2.Forward(activation1.output)
	activation2.Forward(dense2.output)

	fmt.Println(mat.Formatted(activation2.output.Slice(0, 5, 0, activation2.output.RawMatrix().Cols)))
}
