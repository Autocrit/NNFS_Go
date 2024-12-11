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

	var output mat.Dense

	output.Mul(inputs, weights.T())

	for i := 0; i < output.RawMatrix().Rows; i++ {
		for j := 0; j < output.RawMatrix().Cols; j++ {
			output.Set(i, j, output.At(i, j)+biases.At(0, j))
		}
	}

	fmt.Println(mat.Formatted(&output))
}
