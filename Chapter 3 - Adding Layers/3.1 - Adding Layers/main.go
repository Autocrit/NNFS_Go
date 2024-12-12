package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	inputs := mat.NewDense(3, 4, []float64{
		1, 2, 3, 2.5,
		2.0, 5.0, -1.0, 2.0,
		-1.5, 2.7, 3.3, -0.8,
	})

	weights := mat.NewDense(3, 4, []float64{
		0.2, 0.8, -0.5, 1.0,
		0.5, -0.91, 0.26, -0.5,
		-0.26, -0.27, 0.17, 0.87,
	})

	biases := mat.NewDense(1, 3, []float64{2, 3, 0.5})

	weights2 := mat.NewDense(3, 3, []float64{
		0.1, -0.14, 0.5,
		-0.5, 0.12, -0.33,
		-0.44, 0.73, -0.13,
	})

	biases2 := mat.NewDense(1, 3, []float64{
		-1, 2, -0.5,
	})

	var layer1_outputs mat.Dense

	r, c := layer1_outputs.Dims()
	layer1_outputs.Mul(inputs, weights.T())
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			layer1_outputs.Set(i, j, layer1_outputs.At(i, j)+biases.At(0, j))
		}
	}

	var layer2_outputs mat.Dense

	r, c = layer2_outputs.Dims()
	layer2_outputs.Mul(&layer1_outputs, weights2.T())
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			layer2_outputs.Set(i, j, layer2_outputs.At(i, j)+biases2.At(0, j))
		}
	}

	fmt.Println(mat.Formatted(&layer2_outputs))
}
