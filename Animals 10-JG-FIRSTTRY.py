# Import necassary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sklearn
import numpy as np
np.random.seed(42)
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import shutil
from keras import Sequential
from tensorflow.keras.applications import ResNet152V2
from keras.layers import Dense, GlobalAvgPool2D as GAP, Dropout,MaxPooling2D,BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping

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
print("Class Distribution of Skewed:\n", class_sizes)
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

# Loading the Balanced Animal Dataset
# Geting a list of class names from the sampled data directory
class_names = sorted(os.listdir(balanced_data_path))
# Geting the number of samples in each class
class_sizes = []
for name in class_names:
    # Get the number of samples in the class directory
    class_size = len(os.listdir(os.path.join(balanced_data_path, name)))
    class_sizes.append(class_size)
    
# Printing the balanced class distribution
print("Class Distribution of Balanced:\n", class_sizes)
print()

# Implementing Data Augementation 
# Increases accuracy by allowing the model to detect altered images of animals correctly
data_generator = ImageDataGenerator( # Initializing image data generator with the specified image transformations and preprocessing
    rescale=1./255, # rescale: normalizes pixel values from 0-255 to 0-1
    horizontal_flip=True, # horizontal_flip: randomly flips images horizontally
    vertical_flip=True, # vertical_flip: randomly flips images vertically
    rotation_range=20, # rotation_range: randomly rotates images by a given range in degrees
    validation_split=0.2) # validation_split: splits the data into training and validation sets, with 20% of the data used for validation

# Create training data
print("Training data: ")
train_data = data_generator.flow_from_directory( 
    balanced_data_path, # Load training data from the specified directory and apply the generator
    target_size=(256,256), # target_size: resizes the images to a specified size
    class_mode='binary', # class_mode: specifies the type of label encoding, categorical for multiple classes
    batch_size=32, # batch_size: specifies the number of samples per batch
    shuffle=True, # shuffle: shuffles the data after each epoch
    subset='training')  # subset: specifies the subset of data to load, in this case, the Training set
print()

# Create validation data
print("Validation data: ")
valid_data = data_generator.flow_from_directory(
    balanced_data_path, # Load validation data from the specified directory and apply the generator
    target_size=(256,256), 
    class_mode='binary', # class_mode: specifies the type of label encoding, categorical for multiple classes
    batch_size=32, 
    shuffle=True, 
    subset='validation') # subset: specifies the subset of data to load validation data
print()

# Function used to plot images
def show_image(image, image_title=None):
    # Display the image
    plt.imshow(image)
    # Set the title of the plot if provided
    plt.title(image_title)
    # Turn off the axes in the plot
    plt.axis('off')

# Function to grab a random piece of data (image) from the dataset
def get_random_data(data_tuple):
    images, labels = data_tuple
    # geting a random index for an image in the dataset
    idx = np.random.randint(len(images))
    # selecting the image and its corresponding label using the random index
    image, label = images[idx], labels[idx]
    # returning the selected image and label
    return image, label

# Build Deep Learning Model
# Loading the pre-trained ResNet152V2 model, freeze its weights and exclude its final classification layer.
base_model = ResNet152V2(include_top=False, input_shape=(256,256,3), weights='imagenet')
base_model.trainable = False
# Creating a sequential model with the ResNet152V2 base model, a global average pooling layer, two fully connected layers, MaxPooling layer,BatchNormalization layer,and a final softmax classification layer.
sequential_model = Sequential([
    base_model, #Adding ResNet152V2 base model
    MaxPooling2D(pool_size=(2, 2)),  # Adding Max Pooling with a 2x2 window
    BatchNormalization(),  # Adding Batch Normalization
    GAP(), # global average pooling layer to reduce dimension and parameters without losing important features
    Dense(256, activation='relu'), #
    Dropout(0.2), # adding droupout layer to prevent overfitting
    Dense(num_classes, activation='softmax')
], name=name)
# Compile the model with sparse categorical cross-entropy as the loss function, Adam optimizer and accuracy as the evaluation metric.
sequential_model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics=['accuracy']
)

# Train the model using the training and validation datasets, using 50 epochs and the previously defined callbacks.
history = sequential_model.fit(
    train_data, validation_data=valid_data,
    epochs=50, callbacks=[EarlyStopping(patience=3, restore_best_weights=True), ModelCheckpoint(name + ".keras", save_best_only=True)]
)


