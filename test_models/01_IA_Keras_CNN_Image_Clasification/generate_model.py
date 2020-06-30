import cv2
import numpy as np
import random
import os
import sys
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import model_from_json


# funcion de split para el armado de los dataset de 
# entrenamiento y validacion

def split_data(dataset, ratio=0.85):
    index = int(len(dataset) * ratio)
    return dataset[:index], dataset[index:]

# vector de imagenes
images = []
# paths de imagenes
path_with_mask = './test_models/01_IA_Keras_CNN_Image_Clasification/data/train/with_mask'
path_without_mask = './test_models/01_IA_Keras_CNN_Image_Clasification/data/train/without_mask'
list_file_with = os.listdir(path_with_mask)
list_file_without = os.listdir(path_without_mask)
target_dir = './test_models/01_IA_Keras_CNN_Image_Clasification/model/'

# se agregan las imagenes de las caras con barbijos
for filename in tqdm(list_file_with):
    path_img = os.path.join(path_with_mask, filename)
    img_pixel = cv2.imread(path_img, cv2.IMREAD_COLOR) #RBG Level
    # print(path_img)
    img_pixel = cv2.resize(img_pixel, (64, 64)) #64 x 64 pixel
    images.append([np.array(img_pixel), np.array([1, 0])])

# se agregan las imagenes de las caras sin barbijos
for filename in tqdm(list_file_without):
    path_img = os.path.join(path_without_mask, filename)
    img_pixel = cv2.imread(path_img, cv2.IMREAD_COLOR) #RBG Level
    img_pixel = cv2.resize(img_pixel, (64, 64)) #64 x 64 pixel
    images.append([np.array(img_pixel), np.array([0, 1])])

random.shuffle(images)
# train, test = split_data(images)
train = images

# se crean la matrices de datos y labels
train_data = np.array([i[0] for i in train]).reshape(-1, 64, 64, 3)
train_label = np.array([i[1] for i in train])
# test_data = np.array([i[0] for i in test]).reshape(-1, 64, 64, 3)
# test_label = np.array([i[1] for i in test])


# CREATE MODEL

print("Comienzo del proceso de armado del modelos")

# creacion de modelo secuencial
model = Sequential()

# capa input de 64x64 pixeles y 3 dimensiones RGB
model.add(InputLayer(input_shape=[64, 64, 3]))

# primera capa conv+relu+pool
model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

# segunda capa conv+relu+pool
model.add(Conv2D(filters=50, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

# tercera capa conv+relu+pool
model.add(Conv2D(filters=80, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

# dropout y capa plana
model.add(Dropout(0.4))
model.add(Flatten())

# capa de activaacion y salida softmax
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.4))
model.add(Dense(2, activation='softmax'))

# funcion para la optimizacion
optimizer = Adam(learning_rate=1e-4)

# compilacion del modelo
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print("Modelo compilado")

# se muestra la arquitectura del modelo generado
model.summary()

# entrenamiento del modelo
model.fit(x=train_data, y=train_label, epochs=50, batch_size=128, validation_split=0.1)

print("Fin de entrenamiento del modelo")

# impresion de las metricas del modelo craado
scores = model.evaluate(train_data, train_label, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# MODEL TO JSON

# se crea la carpeta el almacenamiento del modelo
if not os.path.exists(target_dir):
  os.mkdir(target_dir)

# creacion y alamacenamiento del modelo en formato jason
model_json = model.to_json()
with open("./test_models/01_IA_Keras_CNN_Image_Clasification/model/model.json", "w") as json_file:
    json_file.write(model_json)

# almacenamiento de los pesos a formato h5
model.save_weights("./test_models/01_IA_Keras_CNN_Image_Clasification/model/weights.h5")
print("Se ha guardado el modelo generado")