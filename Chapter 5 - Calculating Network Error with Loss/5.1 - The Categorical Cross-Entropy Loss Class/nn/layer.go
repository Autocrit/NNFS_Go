package nn

import (
	"gonum.org/v1/gonum/mat"
)

type LayerDense struct {
	weights *mat.Dense
	biases  *mat.Dense
	Output  *mat.Dense
}

func NewLayerDense(inputs, neurons int) *LayerDense {
	layer := LayerDense{
		weights: mat.NewDense(inputs, neurons, rand_array(inputs*neurons)),
		biases:  mat.NewDense(1, neurons, nil),
	}

	return &layer
}

func (layer *LayerDense) Forward(inputs *mat.Dense) {
	layer.Output = mat.NewDense(inputs.RawMatrix().Rows, layer.weights.RawMatrix().Cols, nil)
	layer.Output.Mul(inputs, layer.weights)

	r, c := layer.Output.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			layer.Output.Set(i, j, layer.Output.At(i, j)+layer.biases.At(0, j))
		}
	}
}
