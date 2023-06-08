from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import seaborn as sns
import os
import cv2
import matplotlib.pyplot as plt

train_path="C:\\Users\\Gayathri\\Documents\\Final Year Project\\Tomato Leaf disease detection\\tomato\\train"
val_path="C:\\Users\\Gayathri\\Documents\\Final Year Project\\Tomato Leaf disease detection\\tomato\\val"

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
val_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

val_set = val_datagen.flow_from_directory(val_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical',
                                            shuffle=False)

train_y=training_set.classes
val_y=val_set.classes

print(train_y.shape,val_y.shape)

from tensorflow.keras.models import load_model

MODEL_PATH ='model_tomato_inception.h5'
model = load_model(MODEL_PATH)

y_pred = model.predict(val_set)


import numpy as np
y_pred = np.argmax(y_pred, axis=1)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay
print(classification_report(val_y, y_pred,target_names=training_set.class_indices))


print(accuracy_score(val_y,y_pred))
sns_plot=sns.heatmap(confusion_matrix(val_y,y_pred))
sns_plot = sns_plot.get_figure()
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
sns_plot.savefig("confusion matrix.png")


