package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	inputs := mat.NewVecDense(4, []float64{1, 2, 3, 2.5})
	weights := []*mat.VecDense{
		mat.NewVecDense(4, []float64{0.2, 0.8, -0.5, 1.0}),
		mat.NewVecDense(4, []float64{0.5, -0.91, 0.26, -0.5}),
		mat.NewVecDense(4, []float64{-0.26, -0.27, 0.17, 0.87}),
	}
	biases := mat.NewVecDense(3, []float64{2.0, 3.0, 0.5})
	outputs := mat.NewVecDense(3, nil)

	for i := 0; i < len(weights); i++ {
		outputs.SetVec(i, mat.Dot(inputs, weights[i])+biases.At(i, 0))
	}

	fmt.Println(outputs.RawVector().Data)
}
