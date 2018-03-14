import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost
from sklearn.metrics import mean_squared_error
import lightgbm

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
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

all_data = pd.read_csv('../data/all_data_old_revised1.csv')[0:]
#all_data.describe()
#start_t = 1507
start_t = 1200
all_length = 1820
ts_width = 0
all_data = all_data.fillna(0)[:all_length]
all_data.index = range(len(all_data))
start_time = 365
#trian_data = all_data[:1193]
#test_data = all_data[1193:]

data_feature = all_data[['day_of_week', 'is_holiday', 'around_holiday', 'is_week_end', 'is_work', 'week_no', 'date_old',
                         'is_norm_sunday', 'work_sunday', 'is_saturday', 'is_after_new_year', 'month', 'day_in_month']]


data_cnt = list(all_data['cnt'].values)[start_time: start_t]
# data_cnt = list(map(lambda x: 1 if x == 0 else x, data_cnt))
# data_cnt = list(map(lambda x: np.log(x), data_cnt))


time_series = series_to_supervised(data=data_cnt, n_in=ts_width)


cur_feature = data_feature[start_time: start_t]




pre_feature = data_feature[start_t: all_length]

cur_label = time_series['var1(t)'].values

lgbm = lightgbm.LGBMRegressor(objective='regression', learning_rate=0.04, max_depth=6,
                              n_estimators=200, n_jobs=8, subsample=1)
lgbm.fit(cur_feature, cur_label, verbose=False)
predictions = lgbm.predict(pre_feature)

pre_feature.index = range(len(pre_feature))
result = pd.DataFrame(predictions, columns=['cnt'])
pre_feature['cnt'] = result['cnt']

pre_feature = pre_feature[pre_feature['date_old'] != 0]

pre_feature.loc[pre_feature['cnt'] < 0, 'cnt'] = 40


file_ob = open('sample1.txt', 'w+')
for i in range(len(pre_feature)):
    string = str(int(pre_feature.iloc[i]['date_old'])) + "\t" + str(int(pre_feature.iloc[i]['cnt'])) + "\n"
    file_ob.write(string)



'''

time_series['day_of_week'] = cur_feature['day_of_week']
time_series['is_holiday'] = cur_feature['is_holiday']
time_series['around_holiday'] = cur_feature['around_holiday']
time_series['is_week_end'] = cur_feature['is_week_end']
time_series['is_work'] = cur_feature['is_work']
time_series['week_no'] = cur_feature['week_no']
time_series['date_old'] = cur_feature['date_old']
time_series['is_norm_sunday'] = cur_feature['is_norm_sunday']
time_series['work_sunday'] = cur_feature['work_sunday']
time_series['is_saturday'] = cur_feature['is_saturday']
#time_series['work_saturday'] = cur_feature['work_saturday']
time_series['is_after_new_year'] = cur_feature['is_after_new_year']
time_series['month'] = cur_feature['month']
time_series['day_in_month'] = cur_feature['day_in_month']
#time_series['year'] = cur_feature['year']

time_series.index = range(len(time_series))
label = time_series['var1(t)'].values
time_series.pop('var1(t)')
part = 0.7
split_n = int(len(time_series) * part)
ts_train = time_series[time_series.index <= split_n].values
ts_test = time_series[time_series.index > split_n].values
y_train = label[:len(ts_train)]
y_test = label[len(ts_train):]
# xgb = xgboost.XGBRegressor(n_estimators=120, learning_rate=0.08, subsample=0.8, max_depth=7, n_jobs=8)
# xgb.fit(ts_train, y_train)
# predictions = xgb.predict(ts_test)
lgbm = lightgbm.LGBMRegressor(objective='regression', learning_rate=0.04, max_depth=6,
                              n_estimators=200, n_jobs=8, subsample=1)
lgbm.fit(ts_train, y_train, verbose=False)
predictions = lgbm.predict(ts_test)


mse = mean_squared_error(y_test, predictions)
print(mse)
line1 = plt.plot(range(len(ts_test)), predictions, label=u'predict')
line2 = plt.plot(range(len(y_test)), y_test, label=u'true')
plt.legend()
plt.show()

'''





'''
def rolling(time_series, train_feature, test_feature, day=50, part=0.95):
    column_len = len(time_series.columns)
    result = []
    date = []
    time_series['day_of_week'] = train_feature['day_of_week']
    time_series['is_holiday'] = train_feature['is_holiday']
    time_series['around_holiday'] = train_feature['around_holiday']
    time_series['is_week_end'] = train_feature['is_week_end']
    time_series['is_work'] = train_feature['is_work']
    time_series['week_no'] = train_feature['week_no']
    time_series['date_old'] = train_feature['date_old']
    time_series['is_norm_sunday'] = train_feature['is_norm_sunday']
    time_series['work_sunday'] = train_feature['work_sunday']
    time_series['is_saturday'] = train_feature['is_saturday']
    #time_series['work_saturday'] = train_feature['work_saturday']
    time_series['is_after_new_year'] = train_feature['is_after_new_year']
    time_series['month'] = train_feature['month']
    time_series['day_in_month'] = train_feature['day_in_month']
    time_series['year'] = train_feature['year']
    for i in range(day):
        if i % 25 == 0:
            print(round(i / day * 100, 2), "%")
        # if len(time_series) > window_size:
        #     time_series.drop([0], inplace=True)
        time_series.index = range(len(time_series))
        label = time_series['var1(t)'].values
        split_n = int(len(time_series) * part)
        ts_train = time_series[time_series.index <= split_n].values
        ts_test = time_series[time_series.index > split_n].values
        y_train = label[1:len(ts_train) + 1]
        # xgb = xgboost.XGBRegressor(n_estimators=120, learning_rate=0.08, subsample=0.8, max_depth=7, n_jobs=8)
        # xgb.fit(ts_train, y_train)
        # predictions = xgb.predict(ts_test)
        lgbm = lightgbm.LGBMRegressor(objective='regression', learning_rate=0.04, max_depth=6,
                                      n_estimators=200, n_jobs=8, subsample=1)
        lgbm.fit(ts_train, y_train, verbose=False)
        predictions = lgbm.predict(ts_test)
        
        new_value = predictions[-1]
        if new_value < 0:
            new_value = 0
        result.append(new_value)
        last_row = list(time_series.iloc[-1].values)[1:column_len]
        last_row.append(new_value)
        row_feature = test_feature.iloc[i]
        date.append(row_feature['date_old'])
        last_row.extend(list(row_feature.values))
        new_d = pd.DataFrame([last_row], columns=time_series.columns)
        time_series = time_series.append(new_d, ignore_index=True)
    return time_series, result, date


day = 400
train_feature = data_feature[ts_width: start_t]
test_feature = data_feature[start_t:]


time_series, result, test_date = rolling(time_series, train_feature, test_feature, day=day)



result_DF = pd.DataFrame({'date': test_date, 'result': result})
result_DF = result_DF[result_DF['date'] > 0]
result_DF.index = range(len(result_DF))


#result_DF.to_csv('sample1.csv')





# data1 = pd.read_csv("../../car_platezk_sample.csv")
data2 = train = pd.read_table("../data/train_20171215.txt", engine='python')
group_day_week = data2.groupby(['date', 'day_of_week'], as_index=False)['cnt'].agg({'count1': np.sum})

cnt2 = list(group_day_week['count1'].values)
date2 = list(group_day_week['date'].values)
# cnt1 = list(data1['cnt'].values)
# date1 = list(data1['date'].values)

plt.plot(range(len(cnt2), len(cnt2) + len(result)), result, label=u'predict')
plt.plot(range(len(cnt2)), cnt2, label=u'real')
plt.show()





# date = list(data['date'].values)
# cnt = list(data['cnt'].values)
#


file_ob = open('sample1.txt', 'w+')

for i in range(len(result_DF)):
    string = str(int(result_DF.iloc[i]['date'])) + "\t" + str(int(result_DF.iloc[i]['result'])) + "\n"
    file_ob.write(string)

'''