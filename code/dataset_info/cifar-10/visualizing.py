import pickle 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic

def visualize(arr):
    reshaped_arr = np.array(arr)
    img = reshaped_arr.reshape((32, 32, 3))
    im = Image.fromarray(img)
    im.show()

f = unpickle('./cifar-10/cifar-10/cifar-10-batches-py/data_batch_4')
i = 0

for k,v in f.items():
    if i == 2:   
        #print("The value is ",v)
        for g in range(10):
            visualize(v[g])
        print("The shape of the array is ",np.array(v).shape) 
    #print("The key is ",k)
    i+=1        
print(i)
