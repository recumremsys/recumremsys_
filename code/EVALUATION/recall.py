import os
import pickle
import pandas as pd
from fuzzywuzzy import process

all_files = os.listdir('intersection')
good_files = []

def calculate_similarity(lst):
    new_lst = []
    for obj in lst:
        try:
            similar_obj, score = process.extract(obj, lst)[1]
        except:
            return lst
        #print(obj, similar_obj, score)
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

def check_presence(word, file_name):
    file_name = process.extract(file_name, all_files)[0][0]
    path = 'data/manipulated/' + file_name
    df = pd.read_csv(path)
    text = " ".join(df.iloc[:, 0])
    #print(text)
    if word.lower() in text.lower():
        return True
    else:
        return False

dic = {}
count = 0
total_score = 0
total_matches = 0

for file_name in os.listdir('evaluated_data/ALL'):
    print(file_name)
    intersection = []
    union = []
    csv_file = process.extract(file_name, all_files)[0][0]
    path = 'evaluated_data/ALL/' + file_name
    df = pd.read_csv('ner_objects/' + csv_file)
    model_objects = df.iloc[:, 0].values
    if ".txt" in file_name:
        with open(path, 'r') as fp:
            user_objects = []
            for obj in fp:
                user_objects.append(obj.strip())
    
    elif ".csv" in file_name:
        df1 = pd.read_csv('evaluated_data/ALL/' + csv_file)
        user_objects = df1.iloc[:, 0].values

    model_objects = calculate_similarity(model_objects)
    
    for i in user_objects:
        i = i.lower()
        if not check_presence(i, file_name):
            continue
        if len(model_objects) == 0:
            break
        similar_obj, score = process.extract(i, model_objects)[0]

        if score >= 75:
            intersection.append(i)
            model_objects.remove(similar_obj)
            total_matches += 1
    union = user_objects

    if len(union) == 0:
        total_score += 1
        count += 1
        continue

    score = len(intersection)/len(union)

    if score > 0:
        total_score += score
        good_files.append((file_name, score))
        count += 1

print(count, "of", len(os.listdir('evaluated_data/ALL')))
print(total_score/count)

'''with open('good_files.pkl', 'wb') as fp:
    pickle.dump(good_files, fp)'''

'''
with open('evaluated_data/yash/list_of_items.txt', 'r') as fp:
    dic = {}
    lines = fp.readlines()
    file_name = lines[0].strip()
    for line in lines:
        if "1" in line:
            file_name = line.strip()[2:]
            dic[file_name] = []
            continue
        elif len(line.strip()) != 0:
            dic[file_name].append(line.strip())

for file_name, objects in dic.items():
    file_name  = process.extract(file_name, os.listdir('Name_wise_places_assigned/Evaluation_Data/Yash'))[0][0]
    df = pd.DataFrame()
    df['objects'] = objects
    df.to_csv('evaluated_data/ALL/' + file_name, index = False)'''


with open('good_files.pkl', 'rb') as fp:
    files = pickle.load(fp)

score = 0

for i, j in files:
    score += j
    if j >= 0.50:
        print(i, j)
print(score/len(files))