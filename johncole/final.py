import pandas as pd
import numpy as np
import lightgbm
from sklearn.metrics import mean_squared_error

# train = pd.read_table("../data/fusai_train_20180227.txt", engine='python')
# train = train[["date", "brand", "cnt"]]


# train = train[train['brand'].isin([2, 6, 4, 10])]

# group_day_t = train.groupby(['date'], as_index=False)['cnt'].agg({'count1': np.sum})
d_data = pd.read_csv('../data/train_brand_1/train_brand_1.csv')

all_length = 1800
split_point = 1191
cv_split = 900
# all_data = pd.merge(d_data, group_day_t, on='date', how='left')
all_data = d_data
all_data = all_data.fillna(0)[:all_length]

feature_data = all_data[['week_no', 'day_of_week', 'date', 'lunar_day', 'is_holiday', 'after_holiday_type',
                         'after_holiday', 'before_holiday', 'before_holiday_type', 'before_holiday',
                         'last_workday_before_holiday', 'is_week_end', 'is_work', 'spring_sunday', 'is_norm_sunday',
                         'is_saturday', 'is_after_new_year', 'month', 'day', 'year']]
feature_data.index = range(len(feature_data))

value = list(all_data['cnt_y'].values)
ds = list(all_data['ds'].values)

train_data = feature_data[:split_point]
test_data = feature_data[split_point:all_length]

train_value = value[:split_point]
train_log = np.log(list(map(lambda x: x + 1, train_value)))

test_value = value[cv_split:split_point]


lgbm = lightgbm.LGBMRegressor(objective='regression', learning_rate=0.13, max_depth=6,
                              n_estimators=200, n_jobs=8, subsample=0.8)
lgbm.fit(train_data, train_log, verbose=False)
predictions = lgbm.predict(test_data)
predictions = list(map(lambda x: x - 1, np.exp(predictions)))

test_data.index = range(len(test_data))
test_data['test'] = predictions
test_data['ds'] = ds[split_point: all_length]
train_data['test'] = train_value
train_data['ds'] = ds[:split_point]

td = pd.concat([train_data, test_data], axis=0)
tdd = td[['ds', 'test']]
tdd.index = range(len(tdd))
tdd.loc[tdd['test'] < 0, 'test'] = 0
tdd.to_csv("../data/td.csv")

"""
0.13 6 0.8
"""

# lr = np.arange(0.01, 0.21, 0.01)
# md = np.arange(1, 11, 1)
# ss = np.arange(0.1, 1.01, 0.1)
# mse_min = 10000000
# tp = ()
# for l in lr:
#     for m in md:
#         for s in ss:
#             lgbm = lightgbm.LGBMRegressor(objective='regression', learning_rate=l, max_depth=m,
#                                           n_estimators=200, n_jobs=8, subsample=s)
#             lgbm.fit(train_data, train_log, verbose=False)
#             predictions = lgbm.predict(test_data)
#             mse = mean_squared_error(test_value, list(map(lambda x: x - 1, np.exp(predictions))))
#
#             if mse_min > mse:
#                 mse_min = mse
#                 tp = (l, m, s)
# print(mse_min)
# print(tp)





