import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
from datetime import date

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

def process_categorical(df, column_name):
    df[column_name] = pd.Series(df[column_name], dtype="category")
    return df

def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    return df

def drop_corr(df):
    df = df.drop(["GarageArea", "TotRmsAbvGrd"], axis=1)
    return df

def pre_process(df):
    df = handle_missing_data(df)
    df = process_lot_frontage(df)
    df = process_ages(df)
    df = drop_corr(df)

    for col in ["OverallQual", "OverallCond", "LotFrontage" ]:
        df = process_categorical(df, col)

    dummie_col = ["MSZoning", "Street", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood",
                "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
                "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
                "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "GarageFinish",
                "GarageQual", "GarageType", "GarageCond", "PavedDrive", "SaleType", "SaleCondition"]

    for col in dummie_col:
        df = create_dummies(df, col)
    df = df.drop(dummie_col, axis=1)
    return df

train_data = pre_process(train_data)
test_data = pre_process(test_data)
