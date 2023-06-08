import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
import random
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

IMAGE_SIZE = [224, 224]

train_path = 'C:\\Users\\Gayathri\\Documents\\Final Year Project\\Tomato Leaf disease detection\\tomato\\train'
valid_path = 'C:\\Users\\Gayathri\\Documents\\Final Year Project\\Tomato Leaf disease detection\\tomato\\val'


inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


print("Number of layers in the base model: ", len(inception.layers))


fine_tune_at = 280

for layer in inception.layers[:fine_tune_at]:
  layer.trainable = False    


folders = glob('C:\\Users\\Gayathri\\Documents\\Final Year Project\\Tomato Leaf disease detection\\tomato\\train\\*')


x = Flatten()(inception.output)

prediction = Dense(len(folders), activation='softmax')(x)


model = Model(inputs=inception.input, outputs=prediction)


model.summary()


model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('C:\\Users\\Gayathri\\Documents\\Final Year Project\\Tomato Leaf disease detection\\tomato\\train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('C:\\Users\\Gayathri\\Documents\\Final Year Project\\Tomato Leaf disease detection\\tomato\\val',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical',
                                            shuffle=False)

class_dict = training_set.class_indices
print(class_dict)

es=EarlyStopping(monitor='val_accuracy',mode='max',verbose=1,patience=5)

r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=40,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set),
  callbacks=[es]
)



import matplotlib.pyplot as plt


plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('LossVal_loss_inception')
plt.show()


plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('AccVal_acc_inception')
plt.show()




from tensorflow.keras.models import load_model

model.save('model_tomato_inception.h5')


y_pred = model.predict(test_set)


import numpy as np
y_pred = np.argmax(y_pred, axis=1)



from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model=load_model('model_tomato_inception.h5')


img=image.load_img('C:\\Users\\Gayathri\\Documents\\Final Year Project\\Tomato Leaf disease detection\\tomato\\val\\Tomato___Leaf_Mold\\0a555f63-bf03-4958-8993-e1932b8dce9f___Crnl_L.Mold 9064.jpg',target_size=(224,224))

x=image.img_to_array(img)


x=x/255

import numpy as np
x=np.expand_dims(x,axis=0)


a=np.argmax(model.predict(x), axis=1)

print(a)

