import pandas as pd
import os


length = 600
file_dict = {}
for f in os.listdir("../data/tb2"):
    file_dict[f] = pd.read_csv("../data/tb2/" + f)

file_dict = dict(sorted(file_dict.items(), key=lambda x: int(x[0].split(".")[0])))
# print(file_dict.keys())

table = pd.DataFrame()
first = True
for key, value in file_dict.items():
    if first:
        table["date"] = file_dict[key]["date"]
    k = key.split(".")[0]
    table[k] = file_dict[key]["cnt"]
    table[k + "fill"] = file_dict[key]["brand_fill"]
table.describe()

brand = list(range(1, 11))

file_ob = open('../data/sample3.txt', 'w+')
for i in range(len(table)):
    row = table.iloc[i]
    date = row["date"]
    for b in brand:
        if row[str(b) + "fill"] != 0:
            string = str(int(date)) + "\t" + str(b) + "\t" + str(int(row[str(b)])) + "\n"
            file_ob.write(string)