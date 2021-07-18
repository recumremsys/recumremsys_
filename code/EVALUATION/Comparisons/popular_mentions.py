import os
import ast
import pandas as pd
from fuzzywuzzy import process

def calculate_similarity(lst):
    new_lst = []
    for obj in lst:
        try:
            similar_obj, score = process.extract(obj, lst)[1]
        except:
            return lst
        large = obj
        smaller = similar_obj
        if len(similar_obj) > len(obj):
            large = similar_obj
            smaller = obj
        if score > 80 or smaller in large:
            new_lst.append(smaller)
            continue

        new_lst.append(obj)

    return list(set(new_lst))

average_score = 0
count = 0

for file_name in os.listdir('intersection'):
    score = 0
    objects = pd.read_csv('intersection/' + file_name)
    objects = objects.iloc[:, 0].values

    if len(objects) == 0:
        continue

    try:
        popular_mentions = pd.read_csv('popular_mentions/' + file_name)
    except:
        continue

    popular_mentions = ast.literal_eval(popular_mentions.iloc[:, 1].values[0])
    for mention in popular_mentions[7:]:
        obj, confidence = process.extract(mention, objects)[0]
        if confidence >= 85:
            score += 1

    score = score/len(popular_mentions)
    print(file_name, score)
    average_score += score
    count += 1

print(average_score/count)