import os
import pandas as pd
from fuzzywuzzy import process

def check_detection(obj, lst):
    item, score = process.extract(obj, lst)[0]
    #print(obj, item, score)
    if score > 0.7:
        return True
    else:
        return False

df = pd.read_csv('evaluation_object_detection.csv', index_col=0)

user_objects = list(df.iloc[:, 1].values)
model_objects = list(df.iloc[:, 2].values)

accuracy = 0

for index in range(len(user_objects)):
    score = 0
    user_list = str(user_objects[index]).strip().split(',')
    model_list = str(model_objects[index]).strip().split(',')

    if user_list[0] == 'nan':
        print("EMPTY")
        accuracy += 1
        continue

    for obj in user_list:
        if check_detection(obj, model_list):
            score += 1
    
    accuracy += score/len(user_list)
    print(score/len(user_list))


print(accuracy/index)
print(index)