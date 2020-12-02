import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import time
from os import listdir, path


# Converts a rbg image to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def load_data(dir_list, image_size): #input [first_class, second_class, etc] and (image_width, image_height)

    X = []
    y = []
    image_width, image_height = image_size

    for directory in dir_list:
        
        # Output is 0 for no, 1 for yes
        y_val = 0
        if directory.endswith('yes'):
            y_val = 1
        
        for filename in listdir(directory):
            # load the image
            #image = cv2.imread(directory + '\\' + filename)
            image = cv2.imread(path.join(directory, filename), cv2.IMREAD_UNCHANGED)
            #image = crop_brain_contour(image, plot=False)
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_NEAREST)
            
            # Convert image to grayscale if necessary
            if len(image.shape) != 2:
                image = rgb2gray(image)
            
            # normalize values
            image = image / 255.
            
            X.append(image)
            y.append(y_val)
            # append a value of 1 to the target array if the image
            '''
            # is in the folder named 'yes', otherwise append 0.
            if directory[0] == 'yes': #for the time being we only have yes/no directories
                y.append([1])
            else: #other option is no aka directory[1]
                y.append([0])
            '''
            
    X = np.array(X)
    y = np.array(y)

    # Shuffle the data for good measure
    X, y = shuffle(X, y)

    #print(f'Number of examples is: {len(X)}')
    #print(f'X shape is: {X.shape}')
    #print(f'y shape is: {y.shape}')

    return X, y

# Split dataset (80% train, 20% test)
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, y_train, X_test, y_test

def build_model(input_shape): # tuple in the shape of  model input ex: (image_width, image_height, #channels)

    # Define the input placeholder as a tensor with shape input_shape.
    X_input = Input(input_shape)  # shape=(?, 240, 240, 3)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((2, 2))(X_input)  # shape=(?, 244, 244, 3)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)  # shape=(?, 238, 238, 32)

    # MAXPOOL
    X = MaxPooling2D((4, 4), name='max_pool0')(X)  # shape=(?, 59, 59, 32)

    # MAXPOOL
    X = MaxPooling2D((4, 4), name='max_pool1')(X)  # shape=(?, 14, 14, 32)

    # FLATTEN X
    X = Flatten()(X)  # shape=(?, 6272)
    # FULLYCONNECTED
    X = Dense(1, activation='sigmoid', name='fc')(X)  # shape=(?, 1)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='BrainDetectionModel')

    return model

dir_list = ['TumorIdentifier/no', 'TumorIdentifier/yes']
image_size = (128, 128)

# Load and split data
X, y = load_data(dir_list, image_size)
X_train, y_train, X_test, y_test = split_data(X, y)

model = build_model((image_size[0],image_size[1],1))
model.summary()

