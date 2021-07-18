#preprocess
from config import *

class preprocessing():
    
    def __init__(self,file_name):
        self.file_name = file_name
    
    def pickle_load(self):
        f = pickle.load(open(self.file_name,'rb'))
        return f

    def process(self):
        f = self.pickle_load()
        train_x,train_y = [],[]
        for i in range(len(f)):  
            train_x.append(np.resize(np.array(tf.keras.preprocessing.image.img_to_array(f[i][0])),[60,60,3]))
            #print("The type of the f[i][1] is {} and the value is {} ".format(type(f[i][1]),f[i][1]))
            train_y.append(tf.one_hot(f[i][1],17))
        return np.array(train_x),(np.array(train_y)) 

#x = preprocessing('/content/drive/My Drive/Reminder2/Fashion_augmented_dataset/fashion_data_class.pickle')
x = preprocessing('./data/Object_Detection/fashion_dataset/fashion_data_class.pickle')
train_x,train_y = x.process()
#print("The shape of the train_x is ",train_x.shape)
