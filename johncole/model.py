import os
import pandas as pd
import numpy as np
import xgboost
from sklearn.metrics import mean_squared_error
# from sklearn import model_selection
from matplotlib import pyplot
# from keras.models import Sequential
# import keras.layers as kl

abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
train_file_relative_path = '/data/train_20171215.txt'
testA_path = '/data/test_A_20171225.txt'
testB_path = '/data/test_B_20171225.txt'
train = pd.read_table(abs_path + train_file_relative_path, engine='python')
testA = pd.read_table(abs_path + testA_path, engine='python')
testB = pd.read_table(abs_path + testB_path, engine='python')

all_data = pd.read_csv('../data/holiday_data.csv')[1:]

brands = list(set(train['brand'].values))
days = list(set(train['date'].values))

group_day = train.groupby(['date'], as_index=False)['cnt'].agg({'count1': np.sum})
group_week = train.groupby(['day_of_week'], as_index=False)['cnt'].agg({'cout1': np.sum})
group_brand = train.groupby(['brand'], as_index=False)['cnt'].agg({'count1': np.sum})
group_day_brand = train.groupby(['date', 'brand'], as_index=False)['cnt'].agg({'count1': np.sum})
group_day_week = train.groupby(['date', 'day_of_week'], as_index=False)['cnt'].agg({'count1': np.sum})

x = group_day_week.drop(['count1'], axis=1).values
y = group_day_week['count1'].values
X = np.array(x).reshape((len(y), -1))

# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, shuffle=False, test_size=0.1)
#
# pyplot.plot(X_train, y_train)
# #pyplot.show()
#
# X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
# X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# using LSTM to predict
# model = Sequential()
# model.add(kl.LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), dropout=0.1))
#
# model.add(kl.Dense(1, activation="linear"))
# model.compile(loss='mse', optimizer='adam')
# y_predict = model.fit(X_train, y_train, epochs=10000, batch_size=128,
#                       validation_data=(X_test, y_test), verbose=2, shuffle=False)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


ts_width = 650
ts_week = list(group_day_week['day_of_week'].values)[ts_width:]

time_series = series_to_supervised(data=list(y), n_in=ts_width)
start_t = len(time_series)
part = 0.7
label = time_series['var1(t)'].values
time_series.pop('var1(t)')
time_series.index = range(len(time_series))
split_n = int(len(time_series) * part)
ts_train = time_series[time_series.index <= split_n].values
ts_test = time_series[time_series.index > split_n].values
y_train = label[:len(ts_train)]
y_test = label[len(ts_train):]
xgb = xgboost.XGBRegressor(n_estimators=40, learning_rate=0.08, subsample=0.8, max_depth=7, n_jobs=8)
xgb.fit(ts_train, y_train)
predictions = xgb.predict(ts_test)
print(predictions.shape)
mse = mean_squared_error(y_test, predictions)
print(mse)
line1 = pyplot.plot(range(len(ts_test)), xgb.predict(ts_test), label=u'predict')
line2 = pyplot.plot(range(len(y_test)), y_test, label=u'true')
pyplot.legend()
pyplot.show()


# def rolling(time_series, day=50, part=0.9, window_size=650):
#     result = []
#     for i in range(day):
#         if i % 25 == 0:
#             print(round(i / day * 100, 2), "%")
#         if len(time_series) > window_size:
#             time_series.drop([0], inplace=True)
#         time_series.index = range(len(time_series))
#         label = time_series['var1(t)'].values
#         split_n = int(len(time_series) * part)
#         ts_train = time_series[time_series.index <= split_n].values
#         ts_test = time_series[time_series.index > split_n].values
#         y_train = label[1:len(ts_train) + 1]
#         xgb = xgboost.XGBRegressor(n_estimators=120, learning_rate=0.08, subsample=0.8, max_depth=7, n_jobs=1)
#         xgb.fit(ts_train, y_train)
#         predictions = xgb.predict(ts_test)
#         new_value = predictions[-1]
#         result.append(new_value)
#         last_row = list(time_series.iloc[-1].values)[1:]
#         last_row.append(new_value)
#         new_d = pd.DataFrame([last_row], columns=time_series.columns)
#         time_series = time_series.append(new_d, ignore_index=True)
#     return time_series, result
#
#
# time_series, result = rolling(time_series, day=300)
#
# # var_t = time_series['var1(t)']
# #
# #
# # pre = time_series['var1(t)'][start_t:]
#
# A_len = len(testA)
# #B_len = len(testB)
# A_pre = result[0: A_len]
# #B_pre = pre[A_len: A_len + B_len]
#
# testA['cnt'] = A_pre
# #testB['cnt'] = B_pre.values
#
# #testA = testA.append(testB, ignore_index=False)
# testA.index = testA['date']
# testA.to_csv(abs_path + "zk_sample.csv")
#
# pyplot.plot(range(len(y), len(result) + len(y)), result, label=u'predict')
# pyplot.plot(range(len(y)), y, label=u'real')
# pyplot.legend()
# pyplot.show()



