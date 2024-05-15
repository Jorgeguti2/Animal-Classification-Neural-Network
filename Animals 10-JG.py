# Import necassary libraries
import tensorflow as tf
from tensorflow import keras
import sklearn
import numpy as np
np.random.seed(42)
import matplotlib as mpl
import matplotlib.pyplot as plt
import os  # used for interacting with the file system

# giving path to the animals10 image dataset from kaggle 
data_path = 'raw-image'
# Geting a list of all unique class names from the dataset path
class_names = sorted(os.listdir(data_path))
# Counting the number of classes
num_classes = len(class_names)
# Printing the class names and the total number of classes
print("Class Names: \n", class_names)
print("Number of Classes:", num_classes)

