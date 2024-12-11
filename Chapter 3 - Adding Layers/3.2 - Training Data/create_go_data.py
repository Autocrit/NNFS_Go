from spiral_data import spiral_data

# /usr/bin/python3 create_go_data.py > data.go

points = 100
classes = 3

X, y = spiral_data(points, classes)

# Write as Go array
print("rows := " + str(points * classes))
print("cols := 2")
print("data := []float64{")
for v in X:
	print("\t" + str(v[0]) + ", " + str(v[1]) + ",")
print("}")

print("")

#print("var rows = " + str(points * classes))
#print("var cols = 1")
#print("var data = []float64{")
#for v in y:
#    print(str(v) + ", ", end="")
#print("}")