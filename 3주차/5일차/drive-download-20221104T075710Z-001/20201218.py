#역전파

from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

diabetes = load_diabetes()
x = diabetes.data[:, 2]
y = diabetes.target

plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

w = 1.0
b = 1.0
for xi, yi in zip(x, y) :
    yh = xi * w + b
    err = yi - yh
    w_rate = xi
    w = w + w_rate * err
    b = b + 1 * err
    print("w =", w, ", b =", b)

plt.scatter(x, y)
pt1 = (-0.1, -0.1 * w+b)
pt2 = (0.15, 0.15 * w+b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
