package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	inputs := mat.NewVecDense(4, []float64{1, 2, 3, 2.5})
	weights := mat.NewVecDense(4, []float64{0.2, 0.8, -0.5, 1.0})
	bias := 2.0

	outputs := mat.Dot(inputs, weights) + bias

	fmt.Println(outputs)
}
