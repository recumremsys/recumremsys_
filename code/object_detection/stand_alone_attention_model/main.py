from config import *
from model import *
from preprocess import *

#train_x = np.array([[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]],[[8]],[[9]],[[10]]])
'''
train_x = np.ones([10,96,96,1])
train_y = np.array([0,1,0,0,0,1,1,1,1,0])
train_y = tf.one_hot(train_y,2)
classes = 2
'''
#image_shape = train_x[1:]
classes = 17
epochs = 100
batch_size = 20

image_shape = train_x[0].shape
print("The shape of the image is ",image_shape)

img = Input(shape = image_shape)
resnet = ResNet(classes ,image_shape)(img)
resnet = tk.models.Model(inputs = img,outputs = resnet)
print(resnet.summary())
tk.utils.plot_model(resnet,to_file='model.png')
resnet.compile(optimizer = "Adam",loss = 'categorical_crossentropy',metrics=['acc'])
for i in range(epochs):
    resnet.fit(train_x,train_y,batch_size = batch_size)

