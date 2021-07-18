import pandas as pd 
from main import preprocessing_dataset
import imgaug
import numpy as np
import os
import cv2 as cv
import imgaug.augmenters as iaa
from PIL import Image
import pickle 
from tqdm import tqdm

file_path                   = './code/styles.csv'
image_path                  = './dataset/fashion_dataset/images/'
mydataset_path              = './dataset/mydataset/originals/'
dataset_aug_rotate_90       = './dataset/mydataset/aug_data/rotate_90/'
dataset_aug_rotate_180      = './dataset/mydataset/aug_data/rotate_180/'
dataset_aug_crop            = './dataset/mydataset/aug_data/crop/'
dataset_aug_black_and_white = './dataset/mydataset/aug_data/black_and_white/'
dataset_aug_shear           = './dataset/mydataset/aug_data/shear/'

object_list                 = ['Innerwear Vests','Jackets','Sweaters', 'Caps','Sweatshirts', 'Track Pants',
                               'Dresses','Trousers', 'Shorts', 'Jeans', 'Formal Shoes','Sandals', 
                               'Sunglasses', 'Sports Shoes', 'Casual Shoes', 'Shirts','Tshirts'
                              ]
  
df = pd.read_csv(file_path, error_bad_lines=False)
dict_id_with_objects = preprocessing_dataset('articleType',df,object_list)

def desired_images(path,dict_id_with_objects):
    
    all_ids = []
    all_id = 0
    for k,v in dict_id_with_objects.items():
        if k in object_list:
            if len(v) > 1200:
                all_ids = v[:1200]
                all_id+=len(all_ids)
            else:
                all_ids = v    
            for image in os.listdir(image_path):
                if int(image.split('.')[0]) in all_ids:
                    img = cv.imread(os.path.join(image_path,image))
                    cv.imwrite(mydataset_path+image.split('.')[0]+'.png',img)
                    
            

    if all_id == len(os.listdir(mydataset_path)):        
        print("The number of images in the folder and number of ids are equal, Congratulations!!")
    else:
        print("The number of images in the folder and number of ids are NOT equal, :(")        
        print("as the number of images in the folder is "+str(len(os.listdir(mydataset_path)))+"and the total number of id was "+str(all_id))

class aug():

    def __init__(self,dataset_path,dict_id_with_objects,object_list,dataset_aug_rotate_180,dataset_aug_rotate_90,dataset_aug_black_and_white,dataset_aug_crop,dataset_aug_shear):
        self.dataset_path = dataset_path
        self.dict_id_with_objects = dict_id_with_objects
        self.object_list = object_list
        self.dataset_aug_rotate_90 = dataset_aug_rotate_90
        self.dataset_aug_rotate_180 = dataset_aug_rotate_180
        self.dataset_aug_crop = dataset_aug_crop
        self.dataset_aug_shear = dataset_aug_shear
        self.dataset_aug_black_and_white = dataset_aug_black_and_white
            
    def augmentation(self):
        for k,v in self.dict_id_with_objects.items():
            l = len(v);print("The length of the value is ",l)
            if k in self.object_list:    
                for i in os.listdir(self.dataset_path):
                    if int(i.split('.')[0]) in v:
                        self.augment_conditions(l,os.path.join(self.dataset_path,i))

    def augment_conditions(self,length_of_images,image):
        l = length_of_images
        number_of_augmentations = 0
        print("The length of l in the loop the value is ",l)
        #print(int(l) >= 200)
        #print(int(l) < 300)
        #print(((int(l) >= 200) and (int(l) < 300)))
        if ((int(l) >= 200) and (int(l) < 300)):
            print("1st condiotion")
            number_of_augmentations = 5
            self.rotate_90(image)
            self.rotate_180(image)
            self.black_and_white(image)
            self.crop(image)
            self.shear(image)
        elif ((int(l) >= 300) and (int(l) < 400)):
            print("2nd condiotion")    
            number_of_augmentations = 4
            self.rotate_90(image)
            self.rotate_180(image)
            self.black_and_white(image)
            self.crop(image)        
        elif ((int(l) >= 400) and (int(l) < 500)):
            print("3rd condiotion")
            number_of_augmentations = 3
            self.rotate_90(image)
            self.rotate_180(image)
            self.black_and_white(image)            
        elif ((int(l) >= 500) and (int(l) < 600)):
            print("4th condiotion")
            number_of_augmentations = 2
            self.rotate_90(image)
            self.rotate_180(image)                
        elif ((int(l) >= 600) and (int(l) < 700)):
            print("5th condition")
            number_of_augmentations = 1
            self.rotate_90(image)            
        elif int(l) > 700:
            print("6th condition")
            number_of_augmentations = 0    

    def rotate_90(self,src):
        img = cv.imread(src)
        image = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        cv.imwrite(self.dataset_aug_rotate_90+src.split('/')[-1],image)

    def rotate_180(self,src):
        img = cv.imread(src)
        image = cv.rotate(img, cv.ROTATE_180)
        cv.imwrite(self.dataset_aug_rotate_180+src.split('/')[-1],image)

    def black_and_white(self,src):
        image = cv.imread(src,0)
        cv.imwrite(self.dataset_aug_black_and_white+src.split('/')[-1],image)

    def crop(self,src):
        sr = cv.imread(src)
        crop = iaa.Crop(percent=(0, 0.3)) 
        crop_image=crop.augment_image(sr)
        cv.imwrite(self.dataset_aug_crop+src.split('/')[-1],crop_image)

    def shear(self,src):
        sr = cv.imread(src)
        shear = iaa.Affine(shear=(0,40))
        shear_image=shear.augment_image(sr)
        cv.imwrite(self.dataset_aug_shear+src.split('/')[-1],shear_image)

class data_in_pickle():
    
    def __init__(self,mydataset_path,dict_id_with_objects,dataset_aug_rotate_180,dataset_aug_rotate_90,dataset_aug_black_and_white,dataset_aug_crop,dataset_aug_shear):
        self.mydataset_path = mydataset_path
        self.dict_id_with_objects = dict_id_with_objects
        self.dataset_aug_rotate_90 = dataset_aug_rotate_90
        self.dataset_aug_rotate_180 = dataset_aug_rotate_180
        self.dataset_aug_crop = dataset_aug_crop
        self.dataset_aug_shear = dataset_aug_shear
        self.dataset_aug_black_and_white = dataset_aug_black_and_white

    def label(self):
        label_dict = {}
        i = 0
        for k,v in self.dict_id_with_objects.items():
            if len(v) > 200:
                label_dict[str(k)] = i
                i+=1
        return label_dict        
                
    def main(self):    
        label_dict = self.label()
        data = []
        paths = [self.dataset_aug_shear,self.dataset_aug_rotate_90,self.dataset_aug_rotate_180,
                 self.mydataset_path,self.dataset_aug_crop,self.dataset_aug_black_and_white
                ]
        print(label_dict)
        for path in paths:  
            for image in tqdm(os.listdir(path)):
                img = Image.open(os.path.join(path,image))
                for k,v in self.dict_id_with_objects.items():
                    #print("The image name is ",image.split('.')[0])
                    #print("THe v is ",v)
                    if int(image.split(".")[0]) in v:
                        label = label_dict[k]
                        temp = img.copy()
                        data.append((temp,label)) 

        with open("./dataset/mydataset/fashion"+path.split("/")[-1]+"_data.pickle","wb") as f:
            pickle.dump(list(data),f)                    
'''
data = data_in_pickle(mydataset_path,dict_id_with_objects,dataset_aug_rotate_180,dataset_aug_rotate_90,dataset_aug_black_and_white,dataset_aug_crop,dataset_aug_shear)                    
data.main()         

desired_images(image_path,dict_id_with_objects)
print([(x,len(y)) for (x,y) in dict_id_with_objects.items()])
augment = aug(mydataset_path,dict_id_with_objects,object_list,dataset_aug_rotate_180,dataset_aug_rotate_90,dataset_aug_black_and_white,dataset_aug_crop,dataset_aug_shear)
augment.augmentation()
'''
train_data = pickle.load(open('./dataset/mydataset/fashion_data.pickle','rb'))
print("The length of the training data is ",len(train_data))
