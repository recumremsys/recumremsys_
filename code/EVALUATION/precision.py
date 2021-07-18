import os
import pandas as pd
from fuzzywuzzy import process

count = 0
total_score = 0
files = []
all_files = os.listdir('intersection')
df1 = pd.read_csv('all_objects_clubbed.csv')
complete_list = list(df1.iloc[:, 0].values)
labels = df1.iloc[:, 1].values
precision = 0

for file_name in os.listdir('intersection'):
    file_name = process.extract(file_name, all_files)[0][0]
    score = 0
    path = 'intersection/' + file_name
    df = pd.read_csv(path)
    objects = df.iloc[:, 0].values
    if len(objects) == 0:
        continue
    print("count:-", count)
    count += 1
    for obj in objects:
        obj = process.extract(obj, complete_list)[0][0]
        label = labels[complete_list.index(obj)]
        if label == 1:
            score += 1

    precision += score/len(objects)
    print(score/len(objects))
    

print("Average Precision:- ", precision/count)