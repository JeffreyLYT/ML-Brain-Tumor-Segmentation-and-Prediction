import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import sklearn
import sklearn.cluster         # For KMeans class
import sklearn.mixture         # For GaussianMixture class
import sklearn.preprocessing   # For scale function
import mpl_toolkits.mplot3d    # For enabling projection='3d' feature in Matplotlib

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

image = plt.imread('TumorIdentifier/yes/Y1.jpg')
gray = rgb2gray(image)
plt.imshow(gray, cmap='gray')
plt.show(block=False)
input('press <ENTER> to continue')
print(gray.shape)
print(gray)
