import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Settings
pd.set_option('display.max_rows', None)

# Read data
train_data = pd.read_csv("Data/train.csv")
test_data = pd.read_csv("Data/test.csv")

def drop_columns(df):
    df = df.drop(["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"], axis=1)
    return df

def fill_missing_data(df):
    for col in df.columns:
        if df[col].isna().sum() > 0:
            # print(df[col].dtype)
            if df[col].dtype == "object":
                df[col] = df[col].ffill().bfill()
            else:
                df[col] = df[col].fillna(df[col].mean())
    return df

def generate_new_columns(df):
    df["TotalSF"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"]
    df['TotalPorchSF'] = (df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])

    df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    df['HasGarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    df['HasBsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    df['HasFireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    df['Age'] = 2021 - df['YearBuilt']

    return df

train_data = drop_columns(train_data)
test_data = drop_columns(test_data)

train_data = fill_missing_data(train_data)
test_data = fill_missing_data(test_data)

train_data = generate_new_columns(train_data)
test_data = generate_new_columns(test_data)


train_data_corr = train_data.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(train_data_corr, vmax=.8, square=True)
# plt.savefig("Plots/heatmap.png")

corr_columns = []
for i in train_data_corr:
    if train_data_corr[i]['SalePrice'] > 0.3:
        innerName = train_data_corr[i].name
        if innerName != 'SalePrice':
            corr_columns.append(innerName)

# print(corr_columns)
# print(train_data.info())

df = pd.get_dummies(train_data["MSZoning"])
train_data = pd.concat([train_data, df], axis=1)
cat_colm_train = list(df.columns)
df = pd.get_dummies(train_data["Neighborhood"])
train_data = pd.concat([train_data, df], axis=1)
cat_colm_train = cat_colm_train + list(df.columns)

df = pd.get_dummies(test_data["MSZoning"])
test_data = pd.concat([test_data, df], axis=1)
cat_colm_test = list(df.columns)
df = pd.get_dummies(test_data["Neighborhood"])
test_data = pd.concat([test_data, df], axis=1)
cat_colm_test = cat_colm_test + list(df.columns)

col = list(set(cat_colm_train).intersection(cat_colm_test))

predict_columns = corr_columns#['LotFrontage', 'OverallQual', 'GrLivArea', 'Age', 'MasVnrArea', 'TotalSF',  'FullBath', 'HasGarage', 'TotalPorchSF', 'HasFireplace']
predict_columns = predict_columns + col

typ = "nn"

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import GridSearchCV

if typ == "for":
    train_x = train_data[predict_columns]
    train_y = train_data["SalePrice"]

    # Tune parameters
    search_params = {
        'n_estimators'      : [23, 24, 25],
        'max_features'      : [i for i in range(1, train_x.shape[1])],
        'n_jobs'            : [1],
        'min_samples_split' : [3, 4, 5, 6],
        'max_depth'         : [22, 23, 24]
    }

    model = GridSearchCV(
        RFR(),
        search_params,
        cv = 5,
        n_jobs = -1,
        verbose=True
    )

    model.fit(train_x, train_y)
    print(model.best_estimator_)

    test_x = test_data[predict_columns]

    predictions = model.predict(test_x)

    holdout_ids = test_data["Id"]
    submission_df = {"Id": holdout_ids,
                     "SalePrice": predictions}
    submission = pd.DataFrame(submission_df)

    submission.to_csv("submissions/submission8.csv", index=False)

if typ == "nn":
    train = train_data.iloc[:900,:]
    validation = train_data.iloc[901:,:]

    train_x = train[predict_columns]
    train_y = train["SalePrice"]

    validation_x = validation[predict_columns]
    validation_y = validation["SalePrice"]
    print(train_x.shape)

    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Dense(1024, activation='relu', input_shape=[51]),
        layers.Dense(1024, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(1),
    ])
    model.compile(
        optimizer='adam',
        loss='mse',
    )
    with tf.device('/gpu:0'):
        history = model.fit(
            train_x, train_y,
            validation_data=(validation_x, validation_y),
            batch_size=256,
            epochs=50,
        )

    test_x = test_data[predict_columns]
    predictions = model.predict(test_x)
    predictions = predictions.reshape(1459)
    holdout_ids = test_data["Id"]
    submission_df = {"Id": holdout_ids,
                     "SalePrice": predictions}
    submission = pd.DataFrame(submission_df)

    submission.to_csv("submissions/submission7.csv", index=False)