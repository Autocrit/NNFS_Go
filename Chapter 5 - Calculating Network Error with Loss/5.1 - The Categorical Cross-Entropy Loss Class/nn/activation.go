package nn

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// ReLU

type ActivationReLU struct {
	Output *mat.Dense
}

func NewActivationReLU() *ActivationReLU {
	return new(ActivationReLU)
}

func (act *ActivationReLU) Forward(inputs *mat.Dense) {
	act.Output = mat.NewDense(inputs.RawMatrix().Rows, inputs.RawMatrix().Cols, nil)

	f := func(_, _ int, x float64) float64 { return max(0.0, x) }
	act.Output.Apply(f, inputs)
}

// Softmax

type ActivationSoftmax struct {
	Output *mat.Dense
}

func NewActivationSoftmax() *ActivationSoftmax {
	return new(ActivationSoftmax)
}

func (act *ActivationSoftmax) Forward(inputs *mat.Dense) {
	r, c := inputs.Dims()
	act.Output = mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		maximum := inputs.At(i, 0)
		for j := 1; j < c; j++ {
			maximum = max(maximum, inputs.At(i, j))
		}
		sum := 0.0
		for j := 0; j < c; j++ {
			act.Output.Set(i, j, math.Exp(inputs.At(i, j)-maximum))
			sum += act.Output.At(i, j)
		}
		for j := 0; j < c; j++ {
			act.Output.Set(i, j, act.Output.At(i, j)/sum)
		}
	}
}
