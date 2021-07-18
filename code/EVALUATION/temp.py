import pandas as pd
import os
from fuzzywuzzy import process

all_files = os.listdir('intersection')
all_objects = []

for file_name in process.extract('newyork', all_files)[:20]:
    df = pd.read_csv('intersection/' + file_name[0])
    all_objects.extend(df.iloc[:, 0].values)

print(all_objects)