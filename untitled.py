#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

PATH = os.path.join("C:/Users/Stijn/Documents/pws tensorflow/data")

train_dir = os.path.join(PATH, 'training')
validation_dir = os.path.join(PATH, 'testing')

train_red_dir = os.path.join(train_dir, 'red')  # directory with our training red pictures
train_yellow_dir = os.path.join(train_dir, 'yellow')  # directory with our training yellow pictures
train_green_dir = os.path.join(train_dir, 'green')  # directory with our training green pictures
train_unknown_dir = os.path.join(train_dir, 'unknown')  # directory with our training unknown pictures

validation_red_dir = os.path.join(validation_dir, 'red')  # directory with our validation red pictures
validation_yellow_dir = os.path.join(validation_dir, 'yellow')  # directory with our validation yellow pictures
validation_green_dir = os.path.join(validation_dir, 'green')  # directory with our validation green pictures
validation_unknown_dir = os.path.join(validation_dir, 'unknown')  # directory with our validation unknown pictures


# In[ ]:


#understanding data

num_red_tr = len(os.listdir(train_red_dir))
num_yellow_tr = len(os.listdir(train_yellow_dir))
num_green_tr = len(os.listdir(train_green_dir))
num_unknown_tr = len(os.listdir(train_unknown_dir))

num_red_val = len(os.listdir(validation_red_dir))
num_yellow_val = len(os.listdir(validation_yellow_dir))
num_green_val = len(os.listdir(validation_green_dir))
num_unknown_val = len(os.listdir(validation_unknown_dir))

total_train = num_red_tr + num_yellow_tr + num_green_tr + num_unknown_tr
total_val = num_red_val + num_yellow_val + num_green_val + num_unknown_val


print('total training red images:', num_red_tr)
print('total training yellow images:', num_yellow_tr)
print('total training green images:', num_green_tr)
print('total training unknown images:', num_unknown_tr)

print('--') #improves readability

print('total validation red images:', num_red_val)
print('total validation yellow images:', num_yellow_val)
print('total validation green images:', num_green_val)
print('total validation unknown images:', num_unknown_val)

print('--') #improves readability

print("Total training images:", total_train)
print("Total validation images:", total_val)


# In[ ]:


batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
print('variables loaded')


# In[ ]:


train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')


# In[ ]:


sample_training_images, _ = next(train_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    plotImages(sample_training_images[:5])


# In[ ]:




