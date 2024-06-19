# import libraries
import mlflow
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# Machine Learning Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def main(args):
	# TO DO: enable autologging
    # mlflow.autolog()

    # read data
    df = get_data(args.training_data)
    print(df.head())

    # frame as supervised learning
    n_in, n_out = 2, 1
    col_n_in = len(df.columns)*n_in
    reframed = series_to_supervised(df, n_in, n_out)
    reframed.drop(reframed.columns[list(range(col_n_in, col_n_in+len(df.columns)-1))], axis=1, inplace=True)   

    # scale data
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(reframed) 
    scaled_df = pd.DataFrame(scaled, index=reframed.index, columns=reframed.columns)

    # split data
    X, y = scaled_df.drop("BORE_OIL_VOL(t)", axis=1), scaled_df["BORE_OIL_VOL(t)"]
    X_train, X_test, y_train, y_test = train_test_split_time(X, y, n_test_days=730)

    # train model
    model = train_model(X_train, y_train, args.n_estimators, args.min_samples_split, args.min_samples_leaf)

    # evaluate model
    eval_model(model, X_test, y_test)

# function that reads the data
def get_data(path) :
    print("Reading data...")
    df = pd.read_csv(path)
    df["DATEPRD"] = pd.to_datetime(df["DATEPRD"])
    df.set_index("DATEPRD", inplace=True)
    df.drop(["BORE_GAS_VOL", "BORE_WAT_VOL"], axis = 1, inplace=True)
    
    return df

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True) :
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f'{df.columns[j]}(t-{i})') for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(f'{df.columns[j]}(t)') for j in range(n_vars)]
        else:
            names += [(f'{df.columns[j]}(t+{i})') for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg

def train_test_split_time(X, y, n_test_days=365) :
    n_train_days = X.shape[0] - n_test_days
    X_train, X_test = X[:n_train_days], X[n_train_days:]
    y_train, y_test = y[:n_train_days], y[n_train_days:]

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, n_estimators, min_samples_split, min_samples_leaf) :
    model = RandomForestRegressor(random_state=0, n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    model.fit(X_train, y_train)

    return model

def eval_model(model, X_test, y_test) :
    y_hat = model.predict(X_test)
    score = model.score(X_test, y_test)
    mlflow.log_metric('Model Score', score)
    print('Score:', score)
    mae = mean_absolute_error(y_test, y_hat)
    mlflow.log_metric('Mean Absolute Error', mae)
    print('Mean Absolute Error:', mae)
    rmse = sqrt(mean_squared_error(y_test, y_hat))
    mlflow.log_metric('Root Mean Squared Error', rmse) 
    print('Root Mean Squared Error:', rmse)
    r2 = r2_score(y_test, y_hat)
    mlflow.log_metric('R2 Score', r2)
    print('R2:', r2)

    plot_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_hat})
    plot_df.plot(kind='line', figsize=(10, 6))
    plt.title('Actual vs Predicted Data')
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.grid(True)
    mlflow.log_figure(plt.gcf(), "actual_vs_predict.png")
    
def parse_args() :
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--n_estimators", dest='n_estimators',
                        type=int, default=100)
    parser.add_argument("--min_samples_split", dest='min_samples_split',
                        type=int, default=2)
    parser.add_argument("--min_samples_leaf", dest='min_samples_leaf',
                        type=int, default=1)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")