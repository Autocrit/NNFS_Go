package nn

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type LossCategoricalCrossEntropy struct {
}

func NewLossCategoricalCrossEntropy() *LossCategoricalCrossEntropy {
	return &LossCategoricalCrossEntropy{}
}

func (loss *LossCategoricalCrossEntropy) Calculate(output *mat.Dense, y []int) float64 {
	data_losses := loss.Forward(output, y)

	return stat.Mean(data_losses.RawMatrix().Data, nil)
}

func (loss *LossCategoricalCrossEntropy) Forward(y_pred *mat.Dense, y_true []int) *mat.Dense {
	r, c := y_pred.Dims()
	y_pred_clipped := mat.NewDense(r, c, nil)
	f1 := func(_, _ int, v float64) float64 { return clamp(v, 1e7, 1-1e-7) }
	y_pred_clipped.Apply(f1, y_pred)

	correct_confidences := mat.NewDense(1, r, nil)

	if len(y_true) == r {
		for i := 0; i < r; i++ {
			correct_confidences.Set(0, i, y_pred.At(i, y_true[i]))
		}
	} else if len(y_true) == r*c {
		for i := 0; i < r; i++ {
			v := 0.0
			for j := 0; j < c; j++ {
				v += y_pred.At(i, j) * float64(y_true[i*c+j])
			}
			correct_confidences.Set(0, i, v)
		}
	}

	f2 := func(_, _ int, v float64) float64 { return -math.Log(v) }
	correct_confidences.Apply(f2, correct_confidences)

	return correct_confidences
}
