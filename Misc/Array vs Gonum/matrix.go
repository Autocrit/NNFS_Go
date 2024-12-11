package main

type Matrix struct {
	rows, cols int
	data       []float64
}

// Create a new matrix
func NewMatrix(rows, cols int) *Matrix {
	var m Matrix
	m.rows = rows
	m.cols = cols
	m.data = make([]float64, rows*cols)
	return &m
}

// Return value at row i, column j
func (m *Matrix) At(i, j int) float64 {
	return m.data[i*m.cols+j]
}

// Set value at row i, column j
func (m *Matrix) Set(i, j int, v float64) {
	m.data[i*m.cols+j] = v
}

// Dot product of a and b into receiver
func (m *Matrix) Dot(a, b *Matrix) {
	for i := 0; i < a.rows; i++ {
		for j := 0; j < b.cols; j++ {
			v := 0.0
			for k := 0; k < a.cols; k++ {
				v += a.At(i, k) * b.At(k, j)
			}
			m.Set(i, j, v)
		}
	}
}
