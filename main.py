# Title: Stock Price Prediction using Regression & LSTM
# Author: Brendan Teasdale


from math import floor
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

WINDOW_SIZE = 1
PARAMS = [1, 2, 3, 5, 10]  # Alpha values for Ridge Regression
TRAINING_SIZE = 0.8


def load_dataset(file: str) -> pd.DataFrame:
    data = pd.read_csv(Path(__file__).resolve().parent / "data" / file)
    data = data.dropna()
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop(["Open", "High", "Low", "Adj Close"], axis=1)
    return data


def plot_data(data: pd.DataFrame, title: str = None) -> None:
    plt.figure(figsize=[10.24, 8])
    plt.title(title if title else "Generated Plot")
    plt.plot(data["Date"], data["Close"], label="Close")
    plt.plot(data["Date"], data["Predicted"], label="Prediction")
    plt.legend()
    plt.savefig(Path(__file__).resolve().parent / "figs" / f"{title.replace(' ', '')}.png")
    plt.close()


def window_shift(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add a new column to Dataframe input for target price (closing price shifted WINDOW_SIZE days)
    :param data: Dataframe containing Stock Data (Date, Closing Price, Volume)
    :return: Dataframe containing target closing prices for each ground truth closing price.
    """
    data["Target"] = data[["Close"]].shift(-WINDOW_SIZE)
    data = data[:-WINDOW_SIZE]
    return data


def scale_data(data: pd.DataFrame) -> np.ndarray:
    sc = MinMaxScaler(feature_range=(0, 1))
    return sc.fit_transform(data.drop(columns=["Date"]))


def split_data(data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = floor(TRAINING_SIZE * len(data))
    x_train, y_train = data[:split, :], targets[:split]
    x_test, y_test = data[split:, :], targets[split:]
    return x_train, y_train, x_test, y_test


def plot_alphas(x: List[Union[int, float]], y: List[float]):
    plt.figure(figsize=[10.24, 8])
    plt.title("Performance vs Alpha Values")
    plt.plot(x, y)
    plt.xlabel("Alpha")
    plt.ylabel("Scores")
    plt.savefig(Path(__file__).resolve().parent / "figs" / "plot3.png")
    plt.close()


def regression_model(stock_data: pd.DataFrame) -> float:
    stock_data = window_shift(data=stock_data)
    scaled_data = scale_data(data=stock_data)
    features = scaled_data[:, :-1]
    targets = scaled_data[:, -1]

    (x_train, y_train, x_test, y_test) = split_data(data=features, targets=targets)

    best_param = {"score": 0, "alpha": 0}
    scores = []
    for alpha in PARAMS:
        regression = Ridge(alpha)
        regression.fit(x_train, y_train)
        reg_acc = regression.score(x_test, y_test)
        scores.append(reg_acc)
        if reg_acc > best_param["score"]:
            best_param["score"] = reg_acc
            best_param["alpha"] = alpha
    regression = Ridge(alpha=best_param["alpha"])
    regression.fit(x_train, y_train)
    price_pred = regression.predict(features)
    close = list(features[:, 0])

    price_data = {
        "Date": stock_data.iloc[:, stock_data.columns.get_loc("Date")].values,
        "Close": close,
        "Predicted": price_pred,
    }
    price_df = pd.DataFrame(price_data)
    plot_data(data=price_df, title="plot1")
    plot_alphas(x=PARAMS, y=scores)
    return best_param["score"]


def recurrent_net_model(data: pd.DataFrame) -> None:
    training_data = data.copy()
    training_data = scale_data(training_data)
    prices = []
    targets = []

    for i in range(WINDOW_SIZE, len(training_data)):
        prices.append(training_data[i - WINDOW_SIZE : i, 0])
        targets.append(training_data[i, 0])

    (x_train, y_train, x_test, y_test) = split_data(data=np.array(prices), targets=np.array(targets))

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # need to be 3D for LSTM Algo

    inputs = keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))
    x = LSTM(300, return_sequences=True)(inputs)
    x = LSTM(300, return_sequences=True)(x)
    x = LSTM(300, return_sequences=True)(x)
    out = Dense(1, activation="linear")(x)
    model = keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=(1 - TRAINING_SIZE))
    price_pred = model.predict(np.array(prices))

    price_pred = np.asarray(price_pred)
    price_df = data[1:][["Date"]]
    price_df["Close"] = training_data[1:, 0]
    price_df["Predicted"] = price_pred.ravel()
    plot_data(data=price_df, title="plot2")


def dataset_description(data: pd.DataFrame) -> Tuple[List[float], List[float]]:
    print(data.describe().round(5).to_markdown(index=True))

    price = data.iloc[:, data.columns.get_loc("Close")].values
    volume = data.iloc[:, data.columns.get_loc("Volume")].values
    price_correlations = []
    for delay in range(1, 11):
        lagged = price[delay:]
        trimmed = price[:-delay]
        r, p = pearsonr(lagged, trimmed)
        price_correlations.append(r)

    volume_correlations = []
    for delay in range(1, 11):
        lagged = price[delay:]
        trimmed = volume[:-delay]
        r, p = pearsonr(lagged, trimmed)
        volume_correlations.append(r)
    return price_correlations, volume_correlations


def corr_to_md(price_corr: List[float], volume_corr: List[float]) -> None:
    print("\nCorrelatations with price:")
    corr_df = pd.DataFrame()
    corr_df["lag (days out)"] = range(1, 11)
    corr_df["closing price"] = price_corr
    corr_df["volume"] = volume_corr
    print(corr_df.round(5).to_markdown(index=False))


def run():
    stock_data = load_dataset(file="AAPL.csv")
    stock_data = clean_data(data=stock_data)
    recurrent_net_model(data=stock_data)
    regression_model(stock_data=stock_data)

    (price_corr, volume_corr) = dataset_description(data=stock_data)
    corr_to_md(price_corr, volume_corr)


if __name__ == "__main__":
    run()
