import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)

train_data = pd.read_csv("Data/sales_train.csv")
test_data = pd.read_csv("Data/test.csv")

train_data = train_data.drop(["date", "item_price", "date_block_num"], axis=1)

group = train_data.groupby(["item_id", "shop_id"]).sum()
group = group.reset_index()
group["item_cnt_mnt"] = group["item_cnt_day"] / 34
print(group.info())

typ = "nn"
if typ == "nn":
    msk = np.random.rand(len(group)) < 0.8
    train = group[msk]
    validation = group[~msk]

    predict_columns = ["item_id", "shop_id"]
    train_x = train[predict_columns]
    train_y = train["item_cnt_mnt"]

    validation_x = validation[predict_columns]
    validation_y = validation["item_cnt_mnt"]

    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=[2]),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1),
    ])
    model.compile(
        optimizer='adam',
        loss='mse',
    )

    history = model.fit(
        train_x, train_y,
        validation_data=(validation_x, validation_y),
        batch_size=256,
        epochs=10,
    )

    test_x = test_data[predict_columns]
    predictions = model.predict(test_x)
    print(predictions.shape)
    predictions = predictions.reshape(214200)
    holdout_ids = test_data["ID"]
    submission_df = {"ID": holdout_ids,
                     "item_cnt_month": predictions}
    submission = pd.DataFrame(submission_df)

    submission.to_csv("submissions/submission_5.csv", index=False)