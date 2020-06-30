import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json

# Parametros del modelo

length, height = 150, 150
lr = 0.0005
model_path = './test_models/02_IA_Keras_CNN_Image_Clasification/model/model.json'
weights_model_path = './test_models/02_IA_Keras_CNN_Image_Clasification/model/weights.h5'
test_path = './test_models/02_IA_Keras_CNN_Image_Clasification/data/test'

list_test_files = os.listdir(test_path)

print("Carga del modelo de prediccion")

# se obtiene el modelos en formato json
json_file = open(model_path, 'r')
loaded_model = json_file.read()
json_file.close()
cnn = model_from_json(loaded_model)

# se obtienen los pesos
cnn.load_weights(weights_model_path)

# compilacion del modelo con:
#  funcion de perdida categorical_crossentropy
#  funcion Adam con learning rate = 0.0005
#  metricas accuracy
cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])


# SHOW RESULT

fig = plt.figure(figsize=(14, 14))

# se obtienen cinco imagenes de caras sin barbijos
for index, filename in enumerate(list_test_files):
    path_img = os.path.join(test_path, filename)
    y = fig.add_subplot(6, 5, index+6)

    # se obtiene la imagen
    img_pixel = cv2.imread(path_img, cv2.IMREAD_COLOR) #RBG Level
    img_pixel = cv2.resize(img_pixel, (length, height)) #64 x 64 pixel
    
    # se convierte la imagen a matriz y se realiza la evaluacion
    data = img_pixel.reshape(1, length, height, 3)
    model_out = cnn.predict([data])
    
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
