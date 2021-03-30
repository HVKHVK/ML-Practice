import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_frame = pd.read_csv("Data/iris.data", names=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"])

print(data_frame.isnull().sum())
print(data_frame.info())
print(data_frame["Species"].value_counts())

# data_frame[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].hist()
# plt.savefig("Plots/histograms.png")

df1 = data_frame[data_frame["Species"] == "Iris-virginica"]
df2 = data_frame[data_frame["Species"] == "Iris-versicolor"]
df3 = data_frame[data_frame["Species"] == "Iris-setosa"]

# plt.scatter(df1["SepalLengthCm"], df1["SepalWidthCm"], c="red", label="Iris-virginica")
# plt.scatter(df2["SepalLengthCm"], df2["SepalWidthCm"], c="blue", label="Iris-versicolor")
# plt.scatter(df3["SepalLengthCm"], df3["SepalWidthCm"], c="orange", label="Iris-setosa")
# plt.xlabel("Sepal Length")
# plt.ylabel("Sepal Width")
# plt.legend()
# plt.savefig("Plots/Sepal.png")

# plt.scatter(df1["PetalLengthCm"], df1["PetalWidthCm"], c="red", label="Iris-virginica")
# plt.scatter(df2["PetalLengthCm"], df2["PetalWidthCm"], c="blue", label="Iris-versicolor")
# plt.scatter(df3["PetalLengthCm"], df3["PetalWidthCm"], c="orange", label="Iris-setosa")
# plt.xlabel("Petal Length")
# plt.ylabel("Petal Width")
# plt.legend()
# plt.savefig("Plots/Petal.png")

# corr = data_frame.corr()
# fig, ax = plt.subplots(figsize=(10,10))
# sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')
# plt.savefig("Plots/Heatmap.png")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data_frame['Species'] = le.fit_transform(data_frame['Species'])

from sklearn.model_selection import train_test_split
X = data_frame.drop(columns=['Species'])
Y = data_frame['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
print("LR cross validation:", cross_val_score(lr_model, x_train, y_train, cv=5, scoring='accuracy').mean())

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
print("KNN cross validation:", cross_val_score(knn, x_train, y_train, cv=5, scoring='accuracy').mean())

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
print("DTC cross validation:", cross_val_score(dtc, x_train, y_train, cv=5, scoring='accuracy').mean())
