from spiral_data import spiral_data

# /usr/bin/python3 create_go_data.py > spiral_data.go

points = 100
classes = 3

X, y = spiral_data(points, classes)

print("package main\n")
print("import \"gonum.org/v1/gonum/mat\"\n")
print("func SpiralData() (*mat.Dense, *mat.Dense) {")
print("\tX := mat.NewDense(" + str(points*classes) + ", 2, []float64{")
for v in X:
	print("\t\t" + str(v[0]) + ", " + str(v[1]) + ",")
print("\t})\n")

print("\ty := mat.NewDense(" + str(points*classes) + ", 1, []float64{", end="")
for v in y:
    print(str(v) + ", ", end="")
print("})\n\n\treturn X, y\n}")