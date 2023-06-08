from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
import random
import plotly.express as px
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from IPython.display import clear_output as cls

IMAGE_SIZE = [224, 224]

train_path = 'C:\\Users\\Gayathri\\Documents\\Final Year Project\\Tomato Leaf disease detection\\tomato\\train'
valid_path = 'C:\\Users\\Gayathri\\Documents\\Final Year Project\\Tomato Leaf disease detection\\tomato\\val'


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('C:\\Users\\Gayathri\\Documents\\Final Year Project\\Tomato Leaf disease detection\\tomato\\train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
                                                 shuffle=False)

test_set = test_datagen.flow_from_directory('C:\\Users\\Gayathri\\Documents\\Final Year Project\\Tomato Leaf disease detection\\tomato\\val',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical',
                                            shuffle=False)

class_dict = training_set.class_indices
print("The Different Classes are\n")
print(class_dict)
print(" ")
import os
class_names = sorted(os.listdir('C:\\Users\\Gayathri\\Documents\\Final Year Project\\Tomato Leaf disease detection\\tomato\\train\\'))

diseases = os.listdir(train_path)

plants = []
NumberOfDiseases = 0
for plant in diseases:
    if plant.split('___')[0] not in plants:
        plants.append(plant.split('___')[0])
    if plant.split('___')[1] != 'healthy':
        NumberOfDiseases += 1

nums = {}
for disease in diseases:
    nums[disease] = len(os.listdir(train_path + '/' + disease))

index = [n for n in range(10)]
plt.figure(figsize=(10, 3))
plt.bar(index, [n for n in nums.values()], width=0.3)
plt.xlabel('Diseases', fontsize=10)
plt.ylabel('No of images available', fontsize=10)
plt.xticks(index, diseases, fontsize=10, rotation=90)
plt.title('Images per each class of plant disease')


def plot_images(data, class_names):
    
    r, c = 3, 4
    imgLen = r*c
    
    plt.figure(figsize=(20, 15))
    i = 1
    
    for images, labels in iter(data):
        id = np.random.randint(len(images))
        img = tf.expand_dims(images[id], axis=0)
        lab = class_names[np.argmax(labels[id])]
        
        plt.subplot(r, c, i)
        plt.imshow(img[0])
        plt.title(lab)
        plt.axis('off')
        cls()
        
        i+=1
        if i > imgLen:
            break
    plt.show()

plot_images(training_set, class_names)

