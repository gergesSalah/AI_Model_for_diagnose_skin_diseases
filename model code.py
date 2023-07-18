import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os
import keras
import numpy as np
import cv2 as ocv
# from keras.preprocessing import image
import keras.utils as image
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator
from numpy.random import rand, shuffle
from skimage import feature

#
dir = 'D:/myWork/3rd year/smester 2/Data sets/skin diseases/IMG_CLASSES'
classes = ['1. Eczema','2. Melanoma','3. Atopic Dermatitis','4. Basal Cell Carcinoma','5. Melanocytic Nevi']
Data = []
Lables = []
for category in os.listdir(dir):
    #Newdir = 'C:\\Users\\Bahaa\\Desktop\\DataSets\\xray_dataset_covid19\\train\\NORMAL\\'
    #C:\\Users\\DOC\\Desktop\\DataSets\\xray_dataset_covid19\\train\\NORMAL
    newPath = os.path.join(dir,category)
    for img in os.listdir(newPath):
        img_path = os.path.join(newPath,img)
        if 'Thumbs.db' not in img_path:
            # print(img_path)feature.canny(image2)
            Data.append((image.img_to_array(ocv.resize(ocv.imread(img_path,1),(100,100)))))
            Lables.append(classes.index(category))

combined = list(zip(Data,Lables))
shuffle(combined)
Data[:],Lables[:] = zip(*combined)
X_train = np.array(Data)
Y_train = np.array(Lables)
Y_train = np_utils.to_categorical(Y_train)
print(np.shape(X_train))
print(np.shape(Y_train))

print("the data before augmentation :" ,np.shape(X_train) , "there is a problem here")
#Data Augmentation
dataGen = ImageDataGenerator(rotation_range=20,width_shift_range=0.01,height_shift_range=0.01,horizontal_flip=True,vertical_flip=True)
# dataGen.fit(X_train)
# print("the data after augmentation :" ,np.shape(X_train))
#
# model = Sequential()
# IMAGE_WIDTH=100
# IMAGE_HEIGHT=100
# # IMAGE_SIZE = [100,100] #for VGG16
# IMAGE_CHANNELS=3
#
# # vgg = VGG16(input_shape=IMAGE_SIZE+[IMAGE_CHANNELS],weights='imagenet',include_top=False)
# #
# # for layer in vgg.layers:
# #     layer.trainable = False
#
# # model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS)))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.25))
#
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.25))
#
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.25))
#
# # model.add(Conv2D(256, (3, 3), activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# #
# # model.add(Conv2D(256, (3, 3), activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
#
#
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# # model.add(Dropout(0.5))
#
#
# model.add(Dense(5, activation='softmax')) # 5 because we have cat and dog classes
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# #checkpoint & save the model
# file_path = 'my_modeljjj.h5'
# modelcheckpoint = ModelCheckpoint(file_path,monitor='accuracy',verbose=2,save_best_only=True,mode='max')
# callBackList = [modelcheckpoint]
# hist=model.fit(X_train,Y_train,epochs=20,validation_split=0.05,callbacks=callBackList)
# model.summary()
# print(model.evaluate(X_train,Y_train))

#Prediction
#way 1feature.canny(image2)
img = ocv.imread('0_0.jpg',1)
print("img shape : ", img.shape)
img = ocv.resize(img,(100,100))
img = img.reshape(1, 100, 100, 3)
# img = image.img_to_array(img)
# print(np.shape(img))
# print(classes[(np.argmax(model.predict(img)))])




#
#way 2
mm = keras.models.load_model('my_model4.h5')
print(classes[(np.argmax(mm.predict(img)))])
# plt.plot(hist.history['accuracy'])
# plt.plot(hist.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# # summarize history for loss
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#param_number = output_channel_number * (input_channel_number * kernel_height * kernel_width + 1)