package main

import "fmt"

func main() {
	inputs := []float64{1, 2, 3, 2.5}
	weights := [][]float64{
		{0.2, 0.8, -0.5, 1.0},
		{0.5, -0.91, 0.26, -0.5},
		{-0.26, -0.27, 0.17, 0.87},
	}
	biases := []float64{2.0, 3.0, 0.5}

	var layer_outputs []float64

	for i, weight := range weights {
		neuron_output := 0.0
		for i, input := range inputs {
			neuron_output += input * weight[i]
		}
		neuron_output += biases[i]
		layer_outputs = append(layer_outputs, neuron_output)
	}

	fmt.Println(layer_outputs)
}
