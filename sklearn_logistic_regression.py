import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('Book1.csv')
data.head()
plt.scatter(data.studyhours,data.grade,marker='+',color="red")
print("hamza")
x=data[["studyhours"]]
y=data[["grade"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
print(x_test)
lm.fit(x_test,y_test)
print(lm.predict(x_test))
print(lm.score(x_test,y_test))
