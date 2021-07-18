import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

file_path = './data/object_detection/fashion_dataset/styles.csv'
df = pd.read_csv(file_path, error_bad_lines=False)

#df.info()
#print(df.masterCategory.unique())
grouped = df.groupby(['masterCategory','articleType'])
#print(grouped.first())
#print(grouped.groups)
d = {}
for name, group in df.groupby(['masterCategory', 'articleType']):
    #print(name)
    #print(name[0])
    try:
        d[str(name[0])].append(str(name[1]))
    except:
        d[str(name[0])] = [str(name[1])]    

#print(d)
object_list = ['Caps','Hat','Shoe Accessories', 'Sunglasses', 'Umbrellas', 'Water Bottle',
               'Dresses', 'Innerwear Vests', 'Jackets', 'Jeans', 'Lounge Tshirts', 'Nehru Jackets', 
               'Rain Jacket', 'Rain Trousers','Shirts', 'Shorts', 'Sweaters', 'Sweatshirts', 'Swimwear',
               'Track Pants', 'Tracksuits', 'Trousers', 'Tshirts', 'Casual Shoes',  'Formal Shoes', 
               'Sandals', 'Sports Sandals', 'Sports Shoes',  'Body Lotion', 'Body Wash and Scrub', 'Face Moisturisers',
               'Sunscreen', 'Toner'
              ]
print(len(object_list))

def preprocessing_dataset(col_name,df,object_list):
   
    dict_id_with_object = {}
    for obj in object_list:
        dict_id_with_object[obj] = []
    for i in range(len(df)):
        if df.loc[i,col_name] in object_list:
            dict_id_with_object[df.loc[i,col_name]].append(df.loc[i,'id'])

    return dict_id_with_object

dict_id_with_objects = preprocessing_dataset('articleType',df,object_list)

plt.figure(figsize=(20,10))
def graph(x,y,x_label,y_label,title):
    plt.tight_layout()
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.xticks(rotation=90, fontsize = 18)
    plt.legend(title = title)
    for i in range(len(x)):
      plt.text(x[i], y[i], str(y[i]), fontsize=15)
    #plt.show()
    #plt.savefig('./code/fashion_dataset/Figure_1.png', bbox_inches = 'tight')

x,y = [],[]
for k,v in dict_id_with_objects.items():
    x.append(k)
    y.append(len(v))
y,x = zip(*sorted(zip(y,x)))
print([(a,b) for (a,b) in zip(x,y)])
#graph(x,y,'object_name','times_mentioned','Objects with their count in data')
   
