import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import exp
from sklearn.model_selection import train_test_split

data = pd.read_csv("Book1.csv")
data.head()
plt.scatter(data[["studyhours"]], data[["grade"]], marker="+")
x_train, x_test, y_train, y_test = train_test_split(data["studyhours"], data["grade"], test_size=0.2)


def normalize(values):
    return values - values.mean()


def predict(values, x, c):
    return np.array([1 / (1 + exp(-1 * c + -1 * x * t)) for t in values])


def logistic_regression(values, y):
    n = normalize(values)
    x = 0
    c = 0
    learning_rate = 0.001
    iterations = 150
    for i in range(iterations):
        y_predicted = predict(values, x, c)
        derivative_c = -2 * sum((y - y_predicted) * y_predicted * (1 - y_predicted))
        derivative_x = -2 * sum((y - y_predicted) * y_predicted * (1 - y_predicted) * values)
        x = x - learning_rate * derivative_x
        c = c - learning_rate * derivative_c
    return x, c

print(x_test)
print(y_test)
a, b = logistic_regression(x_train, y_train)

print(a, b)
print(x_test)
print(predict(x_test,a,b))
