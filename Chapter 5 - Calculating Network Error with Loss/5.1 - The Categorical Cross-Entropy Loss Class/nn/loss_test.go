package nn

import (
	"fmt"
	"slices"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestLossCategoricalCrossEntropy_Forward(t *testing.T) {
	softmax_outputs := mat.NewDense(3, 3, []float64{
		0.7, 0.1, 0.2,
		0.1, 0.5, 0.4,
		0.02, 0.9, 0.08})

	class_targets := []int{0, 1, 1}
	loss := NewLossCategoricalCrossEntropy()
	correct_confidences := loss.Forward(softmax_outputs, class_targets)
	fmt.Println(mat.Formatted(correct_confidences))
	/*
		want := mat.NewDense(1, 3, []float64{0.7, 0.5, 0.9})

		if !mat.Equal(correct_confidences, want) {
			t.Errorf("LossCategoricalCrossEntropy.Forward() = %v, want %v", correct_confidences, want)
		}
	*/
	class_targets = []int{1, 0, 0, 0, 1, 0, 0, 1, 0}
	correct_confidences = loss.Forward(softmax_outputs, class_targets)
	fmt.Println(mat.Formatted(correct_confidences))
	/*
		if !mat.Equal(correct_confidences, want) {
			t.Errorf("LossCategoricalCrossEntropy.Forward() = %v, want %v", correct_confidences, want)
		}
	*/
}

func TestLossCategoricalCrossEntropy_Calculate(t *testing.T) {
	// Test with values from book page 124
	softmax_outputs := mat.NewDense(3, 3, []float64{
		0.7, 0.1, 0.2,
		0.1, 0.5, 0.4,
		0.02, 0.9, 0.08})

	class_targets := []int{0, 1, 1}

	loss := NewLossCategoricalCrossEntropy()
	loss_value := loss.Calculate(softmax_outputs, class_targets)

	fmt.Println(loss_value)

	want := 0.38506088005216804
	if loss_value != want {
		t.Errorf("LossCategoricalCrossEntropy.Forward() = %v, want %v", loss_value, want)
	}

	class_targets = []int{1, 0, 0, 0, 1, 0, 0, 1, 0}
	loss_value = loss.Calculate(softmax_outputs, class_targets)
	fmt.Println(loss_value)
	if loss_value != want {
		t.Errorf("LossCategoricalCrossEntropy.Forward() = %v, want %v", loss_value, want)
	}
}

func TestLossCategoricalCrossEntropy_Accuracy(t *testing.T) {
	r := 3
	c := 3
	softmax_outputs := mat.NewDense(r, c, []float64{
		0.7, 0.2, 0.1,
		0.5, 0.1, 0.4,
		0.02, 0.9, 0.08})

	class_targets := []int{0, 1, 1}
	//class_targets = []int{1, 0, 0, 0, 1, 0, 0, 1, 0}

	predictions := make([]int, r)

	// argmax
	for i := 0; i < r; i++ {
		idx := 0
		for j := 1; j < c; j++ {
			if softmax_outputs.At(i, j) > softmax_outputs.At(i, idx) {
				idx = j
			}
		}
		predictions[i] = idx
	}

	if len(class_targets) == r*c {
		tmp := make([]int, r)
		for i := 0; i < r; i++ {
			slice := class_targets[i*c : (i+1)*c]
			tmp[i] = slices.Index(slice, 1)
		}
		class_targets = tmp
	}

	acc_sum := 0
	if len(class_targets) == r {
		for i := 0; i < r; i++ {
			if predictions[i] == class_targets[i] {
				acc_sum += 1
			}
		}
	}

	accuracy := float64(acc_sum) / float64(len(predictions))

	fmt.Printf("acc: %f\n", accuracy)
}
