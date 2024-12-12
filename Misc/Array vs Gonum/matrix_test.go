package main

import (
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// Benchmark 1D array representation of a matrix vs mat.Dense
// when calculating dot product

var result float64

var ra = 12 // Matrix a rows
var ca = 16 // Matrix a cols
var rb = ca // Matrix b rows
var cb = 12 // Matrix b cols

func rand_array(n int) []float64 {
	result := make([]float64, n)
	for i, _ := range result {
		result[i] = rand.Float64()
	}
	return result
}

func BenchmarkDot1D(b *testing.B) {
	ma := NewMatrix(ra, ca)
	ma.data = rand_array(ra * ca)

	mb := NewMatrix(rb, cb)
	mb.data = rand_array(rb * cb)

	m := NewMatrix(ra, cb)

	for i := 0; i < b.N; i++ {
		m.Dot(ma, mb)
	}

	result = m.At(0, 0)
}

func BenchmarkMatDense(b *testing.B) {
	ma := mat.NewDense(ra, ca, rand_array(ra*ca))
	mb := mat.NewDense(rb, cb, rand_array(rb*cb))
	mc := mat.NewDense(ra, cb, nil)

	for i := 0; i < b.N; i++ {
		mc.Mul(ma, mb)
	}

	result = mc.At(0, 0)
}
