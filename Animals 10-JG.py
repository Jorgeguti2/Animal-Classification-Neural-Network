# Import necassary libraries
import tensorflow as tf
from tensorflow import keras
import sklearn
import numpy as np
np.random.seed(42)
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

# Loading the Animal Dataset
# giving path to the animals10 image dataset from kaggle 
skewed_data_path = 'raw-img-skewed'
# Geting a list of all unique class names from the dataset path
class_names = sorted(os.listdir(skewed_data_path))
# Counting the number of classes
num_classes = len(class_names)
# Printing the class names and the total number of classes
print()
print("Number of Classes:", num_classes)
print()
print("Class Names: \n", class_names)
print()
# Geting the number of samples in each class (number of images per class)
class_sizes = []
for name in class_names:
    class_size = len(os.listdir(skewed_data_path + "/" + name))
    class_sizes.append(class_size)
# Printing the class distribution
print("Class Distribution:\n", class_sizes)
print()
#  converting lists to dictionary
class_name_size = dict(zip(class_names, class_sizes))

# Handling Data Imbalance
# Seting the path to the directory where the sample balanced data will be saved
balanced_data_path = 'raw-img-balanced'
# Seting the percentage of each class to sample
sample_percent = 0.1
# Looping through each class directory and copy 2000 images or less to the sampled data directory
for class_name in os.listdir(skewed_data_path):
    # Geting the path to the original class directory
    class_path = os.path.join(skewed_data_path, class_name)
    # Geting the path to the sampled class directory
    sampled_class_path = os.path.join(balanced_data_path, class_name)
    # Geting a list of all the image files in the class directory
    image_files = os.listdir(class_path)
    # Calculating the number of images to sample **************
    image_class_size = class_name_size[class_name]
    if image_class_size > 2000:
        num_images = 2000
    else:
        num_images = int(image_class_size)
     # Samplimg the images
    sampled_images = np.random.choice(image_files, size=num_images, replace=False)
    # Copying the sampled images to the sampled class directory
    for image_name in sampled_images:
        src_path = os.path.join(class_path, image_name)
        dst_path = os.path.join(sampled_class_path, image_name)
        shutil.copyfile(src_path, dst_path)
