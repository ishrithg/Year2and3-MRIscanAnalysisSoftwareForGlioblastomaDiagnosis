"""Necessary Imports"""
import matplotlib
import pandas as pd
import imageio
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import seaborn as sns
import pickle
import os
import cv2
import random
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, ElasticNet
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import scale, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from skimage import data, color, exposure, morphology, measure
from skimage.filters import try_all_threshold, threshold_otsu, threshold_local, sobel, gaussian
from skimage.transform import rotate, rescale, resize
from skimage.restoration import inpaint, denoise_tv_chambolle, denoise_bilateral
from skimage.util import random_noise
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage.feature import canny, corner_harris, corner_peaks, Cascade
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


"""Preprocessing of image data and labels"""
class MasterImage(object):
   def __init__(self, PATH='', IMAGE_SIZE=50):
       self.PATH = PATH
       self.IMAGE_SIZE = IMAGE_SIZE
       self.image_data = []
       self.x_data = []
       self.y_data = []
       self.CATEGORIES = []
       self.list_categories = []

   def get_categories(self):
       for path in os.listdir(self.PATH):
           if '.DS_Store' in path:
               pass
           else:
               self.list_categories.append(path)
       print("Found Categories ", self.list_categories, '\n')
       return self.list_categories

   def Process_Image(self):
       try:
           self.CATEGORIES = self.get_categories()
           for categories in self.CATEGORIES:
               train_folder_path = os.path.join(self.PATH, categories)
               class_index = self.CATEGORIES.index(categories)
               for img in os.listdir(train_folder_path):
                   new_path = os.path.join(train_folder_path, img)
                   try:
                       image_data_temp = cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)
                       image_temp_resize = cv2.resize(image_data_temp, (self.IMAGE_SIZE, self.IMAGE_SIZE))
                       self.image_data.append([image_temp_resize, class_index])
                   except:
                       pass
           data = np.asanyarray(self.image_data)

           for x in data:
               self.x_data.append(x[0])  # Get the X_Data
               self.y_data.append(x[1])  # get the label
           X_Data = np.asarray(self.x_data) / (255.0)  # Normalize Data
           Y_Data = np.asarray(self.y_data)
           X_Data = X_Data.reshape(-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1)
           return X_Data, Y_Data
       except:
           print("Failed to run Function Process Image ")

   def pickle_image(self):
       X_Data, Y_Data = self.Process_Image()
       pickle_out = open('X_Data', 'wb')
       pickle.dump(X_Data, pickle_out)
       pickle_out.close()
       pickle_out = open('Y_Data', 'wb')
       pickle.dump(Y_Data, pickle_out)
       pickle_out.close()
       print("Pickled Image Successfully ")
       return X_Data, Y_Data

   def load_dataset(self):
       try:
           X_Temp = open('X_Data', 'rb')
           X_Data = pickle.load(X_Temp)
           Y_Temp = open('Y_Data', 'rb')
           Y_Data = pickle.load(Y_Temp)
           print('Reading Dataset from PIckle Object')
           return X_Data, Y_Data
       except:
           print('Could not Found Pickle File ')
           print('Loading File and Dataset  ..........')
           X_Data, Y_Data = self.pickle_image()
           return X_Data, Y_Data

"""Loading in a training file of Normal Brain scans and scans with Glioblastoma Brain Tumor"""
if __name__ == "__main__":
   path = r"C:\Users\kpr18\Desktop\TRAINING"
   a = MasterImage(PATH=path, IMAGE_SIZE=80)
   X_Data, Y_Data = a.load_dataset()

"""Print statements that display all the data gathered"""
input_shape = X_Data.shape[1:]
print("Here is the shape of the input: ", input_shape)
print("Here is the Image data: ", X_Data)
print("Here is the index for the classification before random shuffling: ", Y_Data)
random.shuffle(Y_Data)
print("Here is the index for the classification after random shuffling: ", Y_Data)
data = ["Normal Brain" if x == 1 else "Brain Tumor" for x in Y_Data]
print("Here is the labels for the shuffled classification: ", data)
values = np.array(data)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print("Here is the One Hot Encoded list of labels: ", onehot_encoded)

"""Deep Learning model with input layer with 10 units, 1 hidden layer, and an output layer.
The model optimizer is adam, the loss is categorical cross entropy, and the metrics used is accuracy.
This specific model is trained first on training data and will be trained later on testing data"""

model = Sequential()
model.add(Conv2D(10, kernel_size=3, activation='relu', input_shape=input_shape))
model.add(MaxPool2D(2))
model.add(Conv2D(10, kernel_size=3, activation='relu'))
model.add(MaxPool2D(2))
model.add(Flatten())
model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
training = model.fit(X_Data, onehot_encoded, epochs=50)

"""Plotting the learning curves"""
sns.set()
history = training.history
plt.plot(history['loss'], label='Training', marker='.')
plt.plot(history['accuracy'], label='Accuracy', marker='.')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curves')
plt.legend()
print("Maximum accuracy: ", max(history['accuracy']))
print("Minimum loss: ", min(history['loss']))
plt.show()