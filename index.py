import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import sklearn
import sklearn.cluster         # For KMeans class
import sklearn.mixture         # For GaussianMixture class
import sklearn.preprocessing   # For scale function
import mpl_toolkits.mplot3d    # For enabling projection='3d' feature in Matplotlib
import cv2
import os

# Converts a rbg image to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Used image dimensions when resizing the image
IMAGE_DIM = (256, 256)

'''
image = cv2.imread('TumorIdentifier/yes/Y1.jpg', cv2.IMREAD_UNCHANGED)
image = cv2.resize(image, IMAGE_DIM, interpolation = cv2.INTER_NEAREST)
gray = rgb2gray(image)
plt.imshow(gray, cmap='gray')
plt.show(block=False)
input('press <ENTER> to continue')
print(gray.shape)
print(gray)
'''

directories = ['TumorIdentifier/no', 'TumorIdentifier/yes']
X = None
y = []

# Go through dataset directories
for directory in directories:
    
    # Output is 0 for no, 1 for yes
    y_val = 0
    if directory.endswith('yes'):
        y_val = 1
        
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".JPG") or filename.endswith(".jpeg"):
            # Extract image data from current file and resize it
            image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, IMAGE_DIM, interpolation=cv2.INTER_NEAREST)
            
            # Convert image to grayscale if necessary
            if len(image.shape) != 2:
                image = rgb2gray(image)
                
            # Add image data to input array   
            if X is None:
                X = np.array([image])
            else:
                X = np.append(X, [image], axis=0)

            # Add output value of the image
            y.append(y_val)
            
        else:
            continue

y = np.array(y)

print(X.shape)
print(y.shape)

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=0)

print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)
        



