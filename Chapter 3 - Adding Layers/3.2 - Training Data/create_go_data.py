from spiral_data import spiral_data

# /usr/bin/python3 create_go_data.py > spiral_data.go

points = 100
classes = 3

X, y = spiral_data(points, classes)

print("package main\n")
print("import \"gonum.org/v1/gonum/mat\"\n")
print("func spiral_data() (*mat.Dense, []int) {")
print("\tX := mat.NewDense(" + str(points*classes) + ", 2, []float64{")
for v in X:
	print("\t\t" + str(v[0]) + ", " + str(v[1]) + ",")
print("\t})\n")

print("\ty := []int{", end="")
for v in y:
    print(str(v) + ", ", end="")
print("}\n\n\treturn X, y\n}")