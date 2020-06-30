import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras as kr
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json


# GET MODEL

# path de imagenes
path_with_mask = './test_models/01_IA_Keras_CNN_Image_Clasification/data/train/with_mask'
path_without_mask = './test_models/01_IA_Keras_CNN_Image_Clasification/data/train/without_mask'

# se obtiene el modelo generado
json_file = open('./test_models/01_IA_Keras_CNN_Image_Clasification/model/model.json', 'r')
loaded_model = json_file.read()
json_file.close()
model = model_from_json(loaded_model)

# se obtienen los pesos
model.load_weights("./test_models/01_IA_Keras_CNN_Image_Clasification/model/weights.h5")
print("Modelo cargado")

# se compila nuevamente el modelo
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# SHOW RESULT
list_file_with = os.listdir(path_with_mask)
list_file_without = os.listdir(path_without_mask)

fig = plt.figure(figsize=(14, 14))

# se obtienen cinco imagenes de caras con barbijos
for index, filename in enumerate(list_file_with[:5]):
    path_img = os.path.join(path_with_mask, filename)
    y = fig.add_subplot(6, 5, index+1)

    # se obtiene la imagen
    img_pixel = cv2.imread(path_img, cv2.IMREAD_COLOR) #RBG Level
    img_pixel = cv2.resize(img_pixel, (64, 64)) #64 x 64 pixel
    
    # se convierte la imagen a matriz y se realiza la evaluacion
    data = img_pixel.reshape(1, 64, 64, 3)
    model_out = model.predict([data])
    
    # se establece la leyenda de salida
    if np.argmax(model_out) == 0:
        str_label = 'pred: With Mask'
    else:
        str_label = 'pred: Without Mask'

    # se muestra la imagen junto con su leyenda
    y.imshow(img_pixel)
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

# se obtienen cinco imagenes de caras sin barbijos
for index, filename in enumerate(list_file_without[:5]):
    path_img = os.path.join(path_without_mask, filename)
    y = fig.add_subplot(6, 5, index+6)

    # se obtiene la imagen
    img_pixel = cv2.imread(path_img, cv2.IMREAD_COLOR) #RBG Level
    img_pixel = cv2.resize(img_pixel, (64, 64)) #64 x 64 pixel
    
    # se convierte la imagen a matriz y se realiza la evaluacion
    data = img_pixel.reshape(1, 64, 64, 3)
    model_out = model.predict([data])
    
    # se establece la leyenda de salida
    if np.argmax(model_out) == 0:
        str_label = 'pred: With Mask'
    else:
        str_label = 'pred: Without Mask'

    y.imshow(img_pixel)
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)


plt.show()
