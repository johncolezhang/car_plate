import pandas as pd
import lightgbm
from sklearn.metrics import mean_squared_error
import xgboost
import numpy as np
import json

"""
    1: 0.03, 2, 0.8 13431.09 0.15 1, 0.8 13594
    2: 0.05, 2, 0.9 10070.78 9493 0.1, 9, 0.5  (0.18 3 0.5 9492 log)
    3: 0.02, 7, 0.6 10255.07 10251 0.17, 8, 0.3 (0.17 8 0.3 12031 log)
    4: 0.05, 5, 1 11262.57 10018 0.16 3 0.7 (0.16 3 0.7 10022)
    5: 0.03, 3, 0.9 32225.13 30818 0.18 8 0.7 (0.18 8 0.7 30815)
    6: 0.06, 3, 1 4145.77 3357 0.2 4 0.6  (0.2 4 0.6 3366)
    7: 0.08, 2, 0.9 12853.56 12103 0.2 1 0.7 (0.2 1 0.7 12103)
    8: 0.08, 5, 0.6 76385.08 56536 0.17 7 0.4 (0.17 7 0.4 56528)
    9: 0.08, 10, 1 193139 169557 0.19 5 0.2 (0.19 5 0.2 169551)
    10: 0.08, 10, 1 24486 11048 0.01 7 0.6 (0.01 7 0.6 11046)
"""
# dict1 = {}
# for i in range(1, 11):
#     # b_data = pd.read_csv('../data/train_brand_1/b.csv')
#     # b_data.loc[pd.isna(b_data['date']), 'cnt'] = 0
#     td = pd.read_csv('../data/td.csv')
#     all_data = pd.read_csv('../data/train_brand_8/train_brand_' + str(i) + '.csv')
#     all_data = pd.merge(all_data, td, on='ds', how='left')
#     #all_data[np.isnan(all_data['cnt_x'])]['brand_fill'] = 0
#     all_data.loc[pd.isna(all_data['cnt_x']), 'brand_fill'] = 0
#
#     validate_point = 1050
#     split_point = 1379
#     all_length = 1800
#     start = 0
#     all_data = all_data[start:]
#     all_data = all_data.fillna(0)[:all_length]
#     all_data.index = range(len(all_data))
#     value = list(all_data['test'].values)
#     vt_value = value[validate_point: split_point]
#     cnty = list(map(lambda x: x + 1, all_data['test'].values))
#     all_data['test'] = np.log(cnty)
#
#     value = np.log(list(map(lambda x: x + 1, value)))
#
#     data_feature = all_data[['week_no', 'day_of_week', 'lunar_day', 'is_holiday', 'after_holiday_type',
#                              'after_holiday', 'before_holiday_type', 'before_holiday', 'last_workday_before_holiday',
#                              'is_week_end', 'is_work', 'spring_sunday', 'test', 'is_norm_sunday', 'is_saturday',
#                              'is_after_new_year', 'month', 'day', 'year', 'date', 'brand_fill']]
#
#     validate_data = data_feature[:validate_point]
#     validate_value = value[:validate_point]
#
#     vt_data = data_feature[validate_point: split_point]
#
#
#
#
#     lr = np.arange(0.1, 0.31, 0.01)
#     md = np.arange(2, 11, 1)
#     # ss = np.arange(0.1, 1.01, 0.1)
#     ss = [.8]
#     mse_min = 1000000
#     tp = ()
#     for l in lr:
#         for m in md:
#             for s in ss:
#                 lgbm = lightgbm.LGBMRegressor(objective='regression', learning_rate=l, max_depth=m,
#                                               n_estimators=200, n_jobs=8, subsample=s)
#                 lgbm.fit(validate_data, validate_value, verbose=False)
#                 predictions = lgbm.predict(vt_data)
#                 mse = mean_squared_error(vt_value, list(map(lambda x: x - 1, np.exp(predictions))))
#
#                 if mse_min > mse:
#                     mse_min = mse
#                     tp = (l, m, s)
#     print(mse_min)
#     print(tp)
#     dict1[i] = str(tp[0]) + " " + str(tp[1]) + " " + str(tp[2]) + " " + str(mse_min)
#
# for k, v in dict1.items():
#     print(k)
#     print(v)
#
# with open('../data/knn_maps_lyl.json', 'w') as f:
#     json.dump(dict1, f)


dict1 = {1: [0.19, 3, 0.8],
         2: [0.17, 6, 0.8],
         3: [0.19, 9, 0.8],
         4: [0.19, 9, 0.8],
         5: [0.17, 10, 0.8],
         6: [0.15, 10, 0.8],
         7: [0.19, 5, 0.8],
         8: [0.21, 4, 0.8],
         9: [0.16, 9, 0.8],
         10: [0.19, 9, 0.8]}


split_point = 1191
all_length = 1800
start = 0

for key, val in dict1.items():
    td = pd.read_csv('../data/td.csv')
    all_data = pd.read_csv('../data/train_brand_1/train_brand_' + str(key) + '.csv')
    all_data = pd.merge(all_data, td, on='ds', how='left')
    all_data.loc[pd.isna(all_data['cnt_x']), 'brand_fill'] = 0
    all_data = all_data[start:]

    all_data = all_data.fillna(0)[:all_length]
    all_data.index = range(len(all_data))

    # cnty = all_data['cnt_y'].values
    cnty = list(map(lambda x: x + 1, all_data['test'].values))
    all_data['test'] = np.log(cnty)

    value = list(all_data['cnt_x'].values)

    value = np.log(list(map(lambda x: x + 1, value)))


    data_feature = all_data[['week_no', 'day_of_week', 'lunar_day', 'is_holiday', 'after_holiday_type',
                             'after_holiday', 'before_holiday_type', 'before_holiday', 'last_workday_before_holiday',
                             'is_week_end', 'is_work', 'spring_sunday', 'test', 'is_norm_sunday', 'is_saturday',
                             'is_after_new_year', 'month', 'day', 'year', 'date', 'brand_fill']]

    train_data = data_feature[:split_point]
    train_label = value[:split_point]


    test_data = data_feature[split_point:all_length]

    lgbm = lightgbm.LGBMRegressor(objective='regression', learning_rate=val[0], max_depth=val[1],
                                  n_estimators=200, n_jobs=8, subsample=val[2])
    lgbm.fit(train_data, train_label, verbose=False)
    predictions = lgbm.predict(test_data)
    predictions = list(map(lambda x: x - 1, np.exp(predictions)))

    test_data.index = range(len(test_data))
    p_value = pd.DataFrame(predictions, columns=['cnt'])
    test_data['cnt'] = p_value['cnt']
    test_data = test_data[test_data['date'] != 0]

    test_data.loc[test_data['cnt'] < 0, 'cnt'] = 20

    test_data = test_data[['date', 'cnt', 'brand_fill']]
    test_data.to_csv("../data/tb2/" + str(key) + ".csv")

