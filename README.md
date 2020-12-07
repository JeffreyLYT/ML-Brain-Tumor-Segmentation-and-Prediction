# ML-Brain-Tumor-Segmentation-and-Prediction
./TumourIdentifier <-- Folder containing first initial dataset from Kaggle
./TumourClassifier <-- Folder containing second initial dataset from Kaggle
./TumourDatasetFinal <-- Folder containing final combined dataset
./Output <-- Output results written from .ipynb files
./Images <-- Set of images used in our Latex report
./LogisticRegressionModel.ipynb <-- notebook to load data and run Logistic Regression Model
./ConvNetTumourIdentifier.ipynb <-- notebook to load data and run Convolutional Neural Network Model

* Install and import numpy, matplotlib, sklearn, openCV (cv2), imutils and tensorflow
* The files should be run in the order: 
    LogisticRegressionModel.ipynb
    ConvNetTumourIdentifier.ipynb
* GPU is not required. 
* Training takes ~2 hours. 
* The report notebook saves files to the "./Output" directory and generates plots.

The TumourIdentifier data was downloaded from https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection
The TumoutClassifier data was downloaded from https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri

For preprocessing and training the dataset we referenced the tutorial found at https://medium.com/@mohamedalihabib7/brain-tumor-detection-using-convolutional-neural-networks-30ccef6612b0
