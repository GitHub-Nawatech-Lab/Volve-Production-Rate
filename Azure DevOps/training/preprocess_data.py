import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

def scaling_data(df) :
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    return scaled_df

def train_test_split_time(X, y, n_test_days=365) :
    n_train_days = X.shape[0] - n_test_days
    X_train, X_test = X[:n_train_days], X[n_train_days:]
    y_train, y_test = y[:n_train_days], y[n_train_days:]
    return X_train, X_test, y_train, y_test