import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

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
    if abs(train_data_corr[i]['SalePrice']) > 0.05:
        innerName = train_data_corr[i].name
        if innerName != 'SalePrice':
            corr_columns.append(innerName)

from scipy import stats

train_data_num = train_data.select_dtypes(include="number")
train_data_oth = train_data.select_dtypes(exclude="number")

z_scores = stats.zscore(train_data_num)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
train_data_num = train_data_num[filtered_entries]

train_data = pd.concat([train_data_num, train_data_oth], axis=1)

train_data = train_data.dropna()
# print(corr_columns)
# print(train_data.info())
# ax = sns.boxplot(x='MSZoning', y='SalePrice', data=train_data, color='#99c2a2')
# plt.savefig("Plots/Mszoning_box.png")
#
# fig, ax = plt.subplots()
# fig.set_size_inches(20, 20)
# ax = sns.boxplot(x='Neighborhood', y='SalePrice', data=train_data, color='#99c2a2')
# ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
# plt.savefig("Plots/Neighborhood.png")

# import statsmodels.api as sm
# from statsmodels.formula.api import ols
#
# model = ols('SalePrice ~ C(MSZoning)', data=train_data).fit()
# anova_table = sm.stats.anova_lm(model, typ=2)
# print(anova_table)
# model = ols('SalePrice ~ C(Neighborhood)', data=train_data).fit()
# anova_table = sm.stats.anova_lm(model, typ=2)
# print(anova_table)

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

typ = "for"

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import GridSearchCV

if typ == "for":
    train_x = train_data[predict_columns]
    train_y = train_data["SalePrice"]

    # Tune parameters
    search_params =[
        {'n_estimators': [40, 50, 60, 70, 80, 90, 100], 'max_features': [30, 35, 40], 'min_samples_split': [3, 5], 'max_depth': [10, 20, 30, 40, 50]},
    ]

    model = GridSearchCV(
        RFR(),
        search_params,
        cv = 5,
        n_jobs =-1,
        scoring='neg_root_mean_squared_error',
        verbose=4
    )

    model.fit(train_x, train_y)
    print(model.best_estimator_)

    test_x = test_data[predict_columns]

    predictions = model.predict(test_x)

    holdout_ids = test_data["Id"]
    submission_df = {"Id": holdout_ids,
                     "SalePrice": predictions}
    submission = pd.DataFrame(submission_df)

    submission.to_csv("submissions/submission19.csv", index=False)

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