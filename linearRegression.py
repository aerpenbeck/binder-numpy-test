import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

model = LinearRegression()
model.fit(x, y)

r_sq = model.score(x, y)
intercept = model.intercept_
slope = model.coef_


def f(t):
    return slope * t + intercept


print(f"coefficient of determination: {r_sq}")
print(f"intercept: {intercept}")
print(f"slope: {slope}")

plt.rcParams.update({"font.size": 16})
plt.axis([0, 60, 0, 40])
plt.scatter(x, y)
plt.xlabel("$x$", fontsize=18)
plt.ylabel("$y$", fontsize=18, rotation=0)

axis = plt.gca()
xmin, xmax = axis.get_xlim()
plt.plot([xmin, xmax], [f(xmin), f(xmax)], color="green")

plt.show()
