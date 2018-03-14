import pandas as pd


# 上班的星期天
def spring_sunday(ds):
    date = pd.to_datetime(ds)
    if date.weekday() == 6 and all_data_fill[pd.to_datetime(all_data_fill['date']) == date]['is_work'].values[0] == 1:
        return 1
    else:
        return 0


# 正常的Sunday
def is_norm_sunday(ds):
    date = pd.to_datetime(ds)
    if date.weekday() == 6 and all_data_fill[pd.to_datetime(all_data_fill['date']) == date]['is_work'].values[0] == 0:
        return 1
    else:
        return 0


# 自己加的
def is_redundent_sunday(ds):
    date = pd.to_datetime(ds)
    if date.weekday() == 6 and all_data_fill[pd.to_datetime(all_data_fill['date']) == date]['date_old'].values[0] == 0:
        return 1
    else:
        return 0


# 上班的saturday
def is_saturday(ds):
    date = pd.to_datetime(ds)
    if date.weekday() == 5 and all_data_fill[pd.to_datetime(all_data_fill['date']) == date]['is_work'].values[0] == 1:
        return 1
    else:
        return 0


def is_in_holiday(ds):
    date = pd.to_datetime(ds)
    if all_data_fill[pd.to_datetime(all_data_fill['date']) == date]['is_holiday'].values[0] == 1:
        return 1
    else:
        return 0


def is_after_new_year(ds):
    date = pd.to_datetime(ds)
    if date.month == 1 and date.day <= 8:
        return 1
    else:
        return 0


def get_month(ds):
    date = pd.to_datetime(ds)
    return date.month


if __name__ == "__main__":
    all_data_fill = pd.read_csv('../data/Book1.csv')
    all_data_fill = all_data_fill.fillna(0)
    all_data_fill['is_norm_sunday'] = all_data_fill['date'].apply(is_norm_sunday)
    all_data_fill['is_redundant_sunday'] = all_data_fill['date'].apply(is_redundent_sunday)
    all_data_fill['work_sunday'] = all_data_fill['date'].apply(spring_sunday)
    all_data_fill['work_saturday'] = all_data_fill['date'].apply(is_saturday)
    all_data_fill['month'] = all_data_fill['date'].apply(get_month)
    all_data_fill.to_csv("../data/Book1.csv")

    #all_data_fill.describe()
