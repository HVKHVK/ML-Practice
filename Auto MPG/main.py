import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

dataset = pd.read_table("Data/auto-mpg.data", names=column_names, na_values="?", comment='\t', sep=" ",
                        skipinitialspace=True)

# print(dataset.isna().sum()) # 6 missing horsepower

dataset = dataset.dropna()

dataset['Origin'] = dataset['Origin'].map(lambda x: {1: 'USA', 2: 'Europe', 3: 'Japan'}.get(x))

dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
# print(dataset.tail())

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration']],
             diag_kind="kde")

# plt.show()
plt.savefig('Output/Images/data_plot.png')

eval_value = 'Weight'

train_stats = train_dataset.describe()
train_stats.pop(eval_value)
train_stats = train_stats.transpose()

train_labels = train_dataset.pop(eval_value)
test_labels = test_dataset.pop(eval_value)


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# print(model.summary())

EPOCHS = 1000

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[early_stop, tfdocs.modeling.EpochDots()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plt.clf()
plotter.plot({'Basic': history}, metric="mae")
plt.ylim([0, 10])
plt.ylabel('MAE ['+eval_value+']')
# plt.show()
plt.savefig('Output/Images/'+eval_value+'1.png')

plt.clf()
plotter.plot({'Basic': history}, metric="mse")
plt.ylim([0, 20])
plt.ylabel('MSE ['+eval_value+'^2]')
# plt.show()
plt.savefig('Output/Images/'+eval_value+'2.png')

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} " +eval_value.format(mae))

test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values ['+eval_value+']')
plt.ylabel('Predictions ['+eval_value+']')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
# plt.show()
plt.savefig('Output/Images/'+eval_value+'3.png')
