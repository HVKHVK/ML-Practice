import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
from sklearn import preprocessing

# Settings
pd.set_option('display.max_rows', None)

# Read data
train_data = pd.read_csv("Data/train.csv")
test_data = pd.read_csv("Data/test.csv")


def handle_missing_data(data):
    data.drop(["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"], axis="columns", inplace=True)
    data["LotFrontage"] = data["LotFrontage"].fillna(value=round(data["LotFrontage"].mean()))
    data.dropna(inplace=True)
    return data

def process_lot_frontage(df):
    cut_points = [0, 50, 75, 100, 1000]
    label_names = ["Low", "Mid", "High", "VeryHigh"]
    df["LotFrontage"] = pd.cut(df["LotFrontage"], cut_points, labels=label_names)
    return df

def process_ages(df):
    df["Age"] = 2021 - df["YearBuilt"]
    df["CurrentAge"] = 2021 - df["YearRemodAdd"]
    df["GarageAge"] = 2021 - df["GarageYrBlt"]
    df["CurrentOwnerAge"] = 2021 - df["YrSold"]
    df = df.drop(["YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold"], axis=1)
    return df

def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    return df

def drop_corr(df):
    df = df.drop(["GarageArea", "TotRmsAbvGrd"], axis=1)
    return df

def select_features(df):
    df = df.drop(["MSSubClass", "TotRmsAbvGrd"], axis=1)
    return df

def separate_result(df):
    train_data = df.drop(["SalePrice"], axis=1)
    test_data_results = df["SalePrice"]

    return train_data, test_data_results

def process_categorical(df, column_name):
    likert_scale = {'NA': -1, 'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
    df[column_name] = df[column_name].map(likert_scale)
    df[column_name].astype("category")
    return df

def process_categorical_2(df, column_name):
    likert_scale = {'NA': -1, 'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3}
    df[column_name] = df[column_name].map(likert_scale)
    df[column_name].astype("category")
    return df

def process_categorical_3(df, column_name):
    likert_scale = {'NA': -1, 'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5}
    df[column_name] = df[column_name].map(likert_scale)
    df[column_name].astype("category")
    return df

def process_categorical_4(df, column_name):
    likert_scale = {'N': 0, 'Y': 1}
    df[column_name] = df[column_name].map(likert_scale)
    df[column_name].astype("category")
    return df

def process_categorical_5(df, column_name):
    likert_scale = {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
    df[column_name] = df[column_name].map(likert_scale)
    df[column_name].astype("category")
    return df

def process_categorical_6(df, column_name):
    likert_scale = {'Con': 0, 'Oth': 0, 'CWD': 0, 'ConLw': 0, 'ConLI': 0, 'ConLD': 0, 'COD': 1, 'New': 2, 'WD': 3}
    df[column_name] = df[column_name].map(likert_scale)
    return df

def train_pre_process(df):
    df = handle_missing_data(df)
    #df = process_lot_frontage(df)
    df = process_ages(df)
    df = drop_corr(df)

    for col in ["OverallQual", "OverallCond"]:
        df[col].astype("category")

    for col in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "KitchenQual", "GarageQual", "GarageCond", "HeatingQC"]:
        df = process_categorical(df, col)

    for col in ["BsmtExposure"]:
        df = process_categorical_2(df, col)

    for col in ["BsmtFinType1", "BsmtFinType2"]:
        df = process_categorical_3(df, col)

    for col in ["CentralAir"]:
        df = process_categorical_4(df, col)

    for col in ["GarageFinish"]:
        df = process_categorical_5(df, col)

    for col in ["SaleType"]:
        df = process_categorical_6(df, col)

    dummie_col = ["MSZoning", "Street", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood",
                "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
                "MasVnrType", "Foundation", "Heating", "Electrical", "Functional", "GarageType", "PavedDrive", "SaleType", "SaleCondition" ]

    for col in dummie_col:
        df = create_dummies(df, col)
    df = df.drop(dummie_col, axis=1)

    return df

def test_pre_process(df):
    df.drop(["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"], axis="columns", inplace=True)
    df["LotFrontage"] = df["LotFrontage"].fillna(value=round(df["LotFrontage"].mean()))
    #df = process_lot_frontage(df)
    df = process_ages(df)
    df = drop_corr(df)

    for col in ["OverallQual", "OverallCond"]:
        df[col].astype("category")

    for col in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "KitchenQual", "GarageQual", "GarageCond", "HeatingQC"]:
        df = process_categorical(df, col)

    for col in ["BsmtExposure"]:
        df = process_categorical_2(df, col)

    for col in ["BsmtFinType1", "BsmtFinType2"]:
        df = process_categorical_3(df, col)

    for col in ["CentralAir"]:
        df = process_categorical_4(df, col)

    for col in ["GarageFinish"]:
        df = process_categorical_5(df, col)

    for col in ["SaleType"]:
        df = process_categorical_6(df, col)

    dummie_col = ["MSZoning", "Street", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood",
                "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
                "MasVnrType", "Foundation", "Heating", "Electrical", "Functional", "GarageType", "PavedDrive", "SaleType", "SaleCondition" ]

    for col in dummie_col:
        df = create_dummies(df, col)
    df = df.drop(dummie_col, axis=1)

    return df

train_data = train_pre_process(train_data)
train_data, train_data_results = separate_result(train_data)
test_data = test_pre_process(test_data)
test_data = test_data.dropna(axis=1)

col_train = train_data.columns
col_test = test_data.columns
col = list(set(col_train).intersection(col_test))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
# score = cross_val_score(lr, train_data[col], train_data_results, cv=10 )
# accuracy = np.mean(score)
# print(accuracy)
lr.fit(train_data[col], train_data_results)
predictions = lr.predict(test_data[col])
print(predictions)

holdout_ids = test_data["Id"]
submission_df = {"Id": holdout_ids,
                 "SalePrice": predictions}
submission = pd.DataFrame(submission_df)

submission.to_csv("submissions/submission1.csv", index=False)