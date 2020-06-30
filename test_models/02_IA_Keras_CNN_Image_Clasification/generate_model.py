 
import sys
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.keras import backend as K


# se cancela cuaquier otro proceso de entrenamiento
K.clear_session()

# carpetas de archivos de entrenamiento, validacion y almacenamiento del modelo
train_data = './test_models/02_IA_Keras_CNN_Image_Clasification/data/train'
validation_data = './test_models/02_IA_Keras_CNN_Image_Clasification/data/validation'
target_dir = './test_models/02_IA_Keras_CNN_Image_Clasification/model/'


# Parametros de Entrenamiento

epochs=20
length, height = 150, 150
batch_size = 32
steps = 1000
validation_steps = 300
Conv1_filters = 32
Conv2_filters = 64
filter1_size = (3, 3)
filter2_size = (2, 2)
pool_size = (2, 2)
classes = 2
lr = 0.0005


# Dataset de entrenamiento
training_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# cambio de escala en valores de pixel
testing_datagen = ImageDataGenerator(rescale=1. / 255)

# generador de imagenes de entrenamiento
training_generator = training_datagen.flow_from_directory(
    train_data,
    target_size=(height, length),
    batch_size=batch_size,
    class_mode='categorical')

# generador de imagenes de validacion
validate_generator = testing_datagen.flow_from_directory(
    validation_data,
    target_size=(height, length),
    batch_size=batch_size,
    class_mode='categorical')

print("Comienzo del proceso de armado del modelos")

# capa Input y conv+relu+pool
cnn = Sequential()
cnn.add(Convolution2D(Conv1_filters, filter1_size, padding ="same", input_shape=(length, height, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=pool_size))

# tercer capa conv+relu+pool
cnn.add(Convolution2D(Conv2_filters, filter2_size, padding ="same"))
cnn.add(MaxPooling2D(pool_size=pool_size))

# cuarta capa plana + activacion relu
cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))

# droupout + capa de salida softmax
cnn.add(Dropout(0.5))
cnn.add(Dense(classes, activation='softmax'))

# compilacion del modelo con:
#  funcion de perdida categorical_crossentropy
#  funcion Adam con learning rate = 0.0005
#  metricas accuracy
cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])

print("Modelo compilado")

# se muestra la arquitectura del modelo generado
cnn.summary()

# entrenamiento del modelo con:
#  generador de imagenes de entrenamiento
#  cantidad de pasos 1000
#  cantidad de epocas 20
#  generador de imagenes de validacion
#  cantidad de pasos para validacion 300
cnn.fit_generator(
    training_generator,
    steps_per_epoch=steps,
    epochs=epochs,
    validation_data=validate_generator,
    validation_steps=validation_steps)

print("Fin de entrenamiento del modelo")

# se crea la carpeta el almacenamiento del modelo
if not os.path.exists(target_dir):
  os.mkdir(target_dir)

# se crea el modelo .json
model_jason = cnn.to_json()

# se almacena el modelo generado en formato json y los pesos en h5
with open("./test_models/02_IA_Keras_CNN_Image_Clasification/model/model.json", "w") as json_file:
    json_file.write(model_jason)

# se almacenan los pesos en formato h5
cnn.save_weights('./test_models/02_IA_Keras_CNN_Image_Clasification/model/weights.h5')

print("Se ha guardado el modelo generado")
