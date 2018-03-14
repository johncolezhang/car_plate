import pandas as pd
import numpy as np
import os

from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
train_file_relative_path = '/data/train_20171215.txt'
train = pd.read_table(abs_path + train_file_relative_path, engine='python')
train.describe()

a = train.groupby(['date', 'day_of_week'], as_index=False)['cnt']

actions1 = train.groupby(['date', 'day_of_week'], as_index=False)['cnt'].agg({'count1': np.sum})

df_train_target = actions1['count1'].values
df_train_data = actions1.drop(['count1'], axis=1).values

# 切分数据（训练集和测试集）
cv = model_selection.ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

print("GradientBoostingRegressor")
for train, test in cv.split(df_train_data):
    gbdt = GradientBoostingRegressor().fit(df_train_data[train], df_train_target[train])
    result1 = gbdt.predict(df_train_data[test])
    print(mean_squared_error(result1, df_train_target[test]))
    print('......')


