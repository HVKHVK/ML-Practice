import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

type = "sk" # sk, tensor

data = pd.read_table("Data/housing.data", delim_whitespace=True, header=None)
data.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

# data.info()
# print ("Description : \n\n", data.describe())
data.hist(figsize=(20,15), color = 'green')
# plt.show()
plt.savefig('Output/Plots/data_plot.png')

x_data = data.drop(data.columns[12], axis = 1)
y_data = data['LSTAT']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.30, random_state=101)
scaler = MinMaxScaler()
scaler.fit(x_train)

x_train = pd.DataFrame(data = scaler.transform(x_train), columns = x_train.columns, index= x_train.index)
x_test = pd.DataFrame(data = scaler.transform(x_test), columns = x_test.columns, index= x_test.index)

if type == "sk":
    rf_regressor = RandomForestRegressor(n_estimators=500, random_state = 0)
    rf_regressor.fit(x_train, y_train)

    y_pred = rf_regressor.predict(x_test)
    p = mean_squared_error(y_test, y_pred)
    print(p ** 0.5)

if type == "tensor":
    CRIM = tf.feature_column.numeric_column('CRIM')
    ZN = tf.feature_column.numeric_column('ZN')
    INDUS = tf.feature_column.numeric_column('INDUS')
    CHAS = tf.feature_column.numeric_column('CHAS')
    NOX = tf.feature_column.numeric_column('NOX')
    RM = tf.feature_column.numeric_column('RM')
    AGE = tf.feature_column.numeric_column('AGE')
    DIS = tf.feature_column.numeric_column('DIS')
    RAD = tf.feature_column.numeric_column('RAD')
    TAX = tf.feature_column.numeric_column('TAX')
    PTRATIO = tf.feature_column.numeric_column('PTRATIO')
    B = tf.feature_column.numeric_column('B')
    MEDV = tf.feature_column.numeric_column('MEDV')

    input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=20, num_epochs=2000, shuffle=True)
    columns = [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, MEDV]
    model = tf.compat.v1.estimator.DNNRegressor(hidden_units=[8, 8, 8, 8, 8], feature_columns=columns)
    model.train(input_fn = input_func, steps = 50000)

    predict_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test, batch_size=20, num_epochs=1, shuffle=False)
    pred_gen = model.predict(predict_input_func)
    predictions = list(pred_gen)

    final_y_preds = []

    for pred in predictions:
        final_y_preds.append(pred['predictions'])

    print(mean_squared_error(y_test, final_y_preds) ** 0.5)
