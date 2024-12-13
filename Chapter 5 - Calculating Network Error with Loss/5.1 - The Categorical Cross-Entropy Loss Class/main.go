package main

import (
	"fmt"
	"nnfs_go/nn"

	"gonum.org/v1/gonum/mat"
)

func main() {
	X, y := spiral_data()

	dense1 := nn.NewLayerDense(2, 3)
	activation1 := nn.NewActivationReLU()

	dense2 := nn.NewLayerDense(3, 3)
	activation2 := nn.NewActivationSoftmax()

	dense1.Forward(X)
	activation1.Forward(dense1.Output)

	dense2.Forward(activation1.Output)
	activation2.Forward(dense2.Output)

	fmt.Println(mat.Formatted(activation2.Output.Slice(0, 5, 0, activation2.Output.RawMatrix().Cols)))

	loss_function := nn.NewLossCategoricalCrossEntropy()
	loss := loss_function.Calculate(activation2.Output, y)

	fmt.Printf("Loss: %f\n", loss)
}
