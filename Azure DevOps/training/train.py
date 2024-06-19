import numpy as np
import pandas as pd
from math import sqrt
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from preprocess_data import series_to_supervised, scaling_data, train_test_split_time

# Split the dataframe into test and train data


def split_data(df):
    """Split a dataframe into training and validation datasets"""
    n_in, n_out = 2, 1
    col_n_in = len(df.columns)*n_in
    reframed = series_to_supervised(df, n_in, n_out)
    reframed.drop(reframed.columns[list(range(col_n_in, col_n_in+len(df.columns)-1))], axis=1, inplace=True)
    scaled_df = scaling_data(reframed)
    X_train, X_test, y_train, y_test = train_test_split_time(scaled_df.drop(scaled_df.columns[-1], axis=1), scaled_df[scaled_df.columns[-1]], n_test_days=730)
    train_data, test_data = (X_train, y_train), (X_test, y_test)

    return (train_data, test_data)


# Train the model, return the model
def train_model(data):
    """Train a model with the given datasets and parameters"""
    # The object returned by split_data is a tuple.
    # Access train_data with data[0] and valid_data with data[1]

    model = RandomForestRegressor(random_state=0).fit(data[0][0], data[0][1])

    return model


# Evaluate the metrics for the model
def get_model_metrics(model, data):
    """Construct a dictionary of metrics for the model"""
    pred = model.predict(data[1][0])
    model_metrics = {
        "Model Score" : model.score(data[1][0], data[1][1]),
        "R2" : r2_score(data[1][1], pred),
        "Mean absolute error" : mean_absolute_error(data[1][1], pred),
        "Root mean squared error" : sqrt(mean_squared_error(data[1][1], pred))
    }
    print(model_metrics)

    return model_metrics
