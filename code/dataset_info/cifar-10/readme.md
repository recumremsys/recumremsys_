The dataset can be downloaded form the [link](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)  
After downloading the dataset convert the batch files with prefix as *.pickle* such as *test_batch.pickle* or *data_batch_2.pikle*    
The pickle file contains the dicitonary which has four keys i.e.  
1. b'batch_label'
2. b'labels'
3. b'data'
4. b'filenames'
   
Each pickle has 1 batch info and total are 5 batches and therefore the batch label is stored in the batch_label   
The shape of the labels is  (10000,)    
The shape of the data is  (10000, 3072)     
The shape of the filenames is  (10000,)    
The dataset the research paper implementation expects is something like                   
*(<PIL.Image.Image image mode=RGB size=32x32 at 0x7F37431E5E48>, label_of_the_image)*
