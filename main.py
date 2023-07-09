import math
import numpy
import numpy as np

arr = np.array(-1)

for i in range(5):
    arr = numpy.vstack((arr, np.array(i)))
# arr.reshape()
arr = np.reshape(arr, (-1,))
e = 9
h = 7

print(math.pi)

