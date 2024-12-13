package nn

import (
	"math/rand/v2"
)

func rand_array(n int) []float64 {
	rdata := make([]float64, n)
	for i := 0; i < n; i++ {
		rdata[i] = rand.NormFloat64() * 0.1
	}
	return rdata
}

func clamp(v, low, high float64) float64 {
	return min(high, max(low, v))
}
