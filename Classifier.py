# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 23:45:16 2018

@author: Bharath Gunari
"""

import os  
from tqdm import tqdm
import cv2
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import csv

TRAIN_DIR = "./train"
IMG_SIZE = 50
TEST_DIR = "./test/"

def CreateDataset():
    counter = 1
    for image in tqdm(os.listdir(TRAIN_DIR)):
        #get the name cat or dog
        label = image.split('.')[-3]
        if label == 'cat':
            GroundTruth = "cats"
        elif label == 'dog':
            GroundTruth = "dogs"
        path = os.path.join(TRAIN_DIR,image)
        #resize the image for 50*50
        image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
        #save the image in respective path
        cv2.imwrite(r"./training_data/"+GroundTruth+"/"+str(counter)+".jpg",image)
        counter +=  1
    
def TestData(model):
    TestimageNmber = {}
    
    for image in tqdm(os.listdir(TEST_DIR)):
        TestImageFileName = image.split('.')
        path = os.path.join(TEST_DIR,image)
        testimage = cv2.imread(path)
        testimage = cv2.resize(testimage,(50,50))
        testimage = testimage.reshape(1,50,50,3)
        TestimageNmber[int(TestImageFileName[0])] = model.predict(testimage)[0][0]

    return TestimageNmber

def WriteResultsIntoCSV(Results):
    Results = dict(sorted(Results.items()))
    print("Writing CSV")
    with open('results.csv','w',newline="") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in tqdm(Results.items()):
            writer.writerow([key, value])
    

def TrainModel():
    # Initialising the CNN
    model = Sequential()    
    # Convolution
    model.add(Conv2D(32, (3, 3), input_shape = (50, 50, 3), activation = 'relu'))

    # Pooling
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Second convolutional layer
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Flattening
    model.add(Flatten())

    # Full connection
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dense(units = 1, activation = 'sigmoid'))

    # Compiling the CNN
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
    #test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory('training_data',target_size = (50, 50),batch_size = 32,class_mode = 'binary')

    model.fit_generator(training_set,steps_per_epoch = 8000,epochs = 25, validation_steps = 2000)
    print("Model Training Done")
    return model


if __name__ == '__main__':
    #Normalize resize the image and seperate the training data
    #CreateDataset()
    #Train the model
    print("Training the Model")
    model = TrainModel()
    print("Testing the Model")
    Results = TestData(model)
    WriteResultsIntoCSV(Results)
    
        