import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import sklearn
import sklearn.cluster         # For KMeans class
import sklearn.mixture         # For GaussianMixture class
import sklearn.preprocessing   # For scale function
import mpl_toolkits.mplot3d    # For enabling projection='3d' feature in Matplotlib

image = plt.imread('TumorIdentifier/yes/Y1.jpg')
print(image.shape)
plt.imshow(image)
plt.show(block=False)
input('press <ENTER> to continue')