# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,notebooks_py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Ejemplo de uso de biblioteca uk-nilm

# %% [markdown]
# ## Celda Colab
# Si se corre en Colab tendra que montar su google drive y seleccionar en ***PATH_DRIVE*** el directorio en el que esta clonado su repositorio

# %%
PATH_DRIVE =  '/content/drive/My Drive/base_de_datos_nilm'
import os
if 'COLAB_GPU' in os.environ:
    print("Estoy corriendo en Colab")
    %tensorflow_version 2.x
    import tensorflow as tf
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('No se ha activado la GPU de Colab')
    print('Found GPU at: {}'.format(device_name))
    from google.colab import drive
    drive.mount('/content/drive')
    %cd $PATH_DRIVE
   
else:
   print("NO estoy corriendo en Colab")

# %% [markdown]
# ## Importo librerias

# %%
import os
if 'COLAB_GPU' in os.environ:
    print("Estoy corriendo en Colab")
    %tensorflow_version 2.x
    import tensorflow as tf
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
      raise SystemError('No se ha activado la GPU de Colab')
    print('Found GPU at: {}'.format(device_name))
else:
    print("NO estoy corriendo en Colab")
    import tensorflow as tf

# %%
import numpy as np
import matplotlib.pyplot as plt
import uk

# %% [markdown]
# ## Cargo parametros

# %%
ELECTRODOMESTICO = 'kettle' #Electrodomestico para el que se entrenara la red
TIPO_RED = 'rectangulos' #Tipo de red a utilizar
NUM_EPOCAS = 20 #numero de epocas de entrenamiento
RUTA_DE_DATOS_X_Y = 'vectores' #Carpeta en la cual se encuentran guardados los vectores X e Y
RUTA_DATOS_ENTRENAMIENTO = 'pesos' #Carpeta en la que se guardaran los datos generados en el entrenamiento

# %% [markdown]
# ## Cargo datos
# Los datos pueden ser generados mediante el notebook "Procesamiento de datos" o bajarlos desde: https://iie.fing.edu.uy/~cmarino/NILM/vectores.zip
#
# Nota: no se garantiza que los datos anteriores se encuentren accesibles en un futuro cercano.

# %%
x_train, y_train = uk.cargar_X_y('data', 'train', ruta_base = RUTA_DE_DATOS_X_Y)
x_validacion, y_validacion = uk.cargar_X_y('data', 'validacion', ruta_base = RUTA_DE_DATOS_X_Y)
# x_test_no_visto, y_test_no_visto = uk.cargar_X_y('data', 'test_no_visto', ruta_base = 'vectores')
# x_test_visto, y_test_visto = uk.cargar_X_y('data', 'test_visto', ruta_base = 'vectores')

# %% [markdown]
# #### Se selecciona el ELECTRODOMESTICO y red a utilizar

# %%
X_train = x_train[TIPO_RED][ELECTRODOMESTICO]
X_val = x_validacion[TIPO_RED][ELECTRODOMESTICO]
X_train = x_train[TIPO_RED][ELECTRODOMESTICO]
X_val = x_validacion[TIPO_RED][ELECTRODOMESTICO]
y_train = y_train[TIPO_RED][ELECTRODOMESTICO]
y_val = y_validacion[TIPO_RED][ELECTRODOMESTICO]

# El expand dims es necesario para entrenar, tensorflow solicita un vector de 3 dimensiones
X_train = np.expand_dims(
    X_train[:, :-3], axis=2
)
X_val = np.expand_dims(
    X_val[:, :-3], axis=2
)

# %% [markdown]
# ### Se normalizan los datos
# Para que el entrenamiento de las redes neuronales sea más efectivo es necesario normalizar los datos. <br>
# El criterio de normalización es el que aparece en la **"Sección 3.6: Standardisation." del paper de Jack Kelly**.
# <br><br>
# **Entrada**: Se le resta la media a cada fila (muestra) y se divide por la media de las desviaciones estandar **de train**.<br>
# **Salida**: Se divide entre el valor máximo en potencia en train, de forma que la salida este en el rango [0,1]. Este calculo depende de que tipo de red se trate
#
#
# NOTA: Los valores utilizados para la normalización solo se calculan para entrenar salvo a la hora de restar la media. Es decir, para predecir valores se debe normalizar con la desviación estándar de train y NO de ella misma.
#

# %%
# Normalizacion de X e y
std_entrada = np.nanmean(np.nanstd(X_train, axis=1))
X_train_norm = uk.utils.normalize_X(X_train, std_entrada)
X_val_norm = uk.utils.normalize_X(X_val, std_entrada)

if TIPO_RED=='rectangulos':
    std_salida = np.max(y_train[:, 0]) ### Para red de rectangulos
else:
    std_salida = np.max(y_train) ### Para red de autoencoder
y_train_norm = uk.utils.normalize_Y(y_train, std_salida, TIPO_RED)
y_val_norm = uk.utils.normalize_Y(y_val, std_salida, TIPO_RED)

# %% [markdown]
# ## Se define el modelo de red neuronal utilizada
# A continuacion se oueden observar los 2 posible modelos, el primero el de rectangulos, el segundo el de autoencoder.
# <img src="img/arquitecturas.jpg"  width="750">

# %%
input_size = X_train.shape[1]

if TIPO_RED=='rectangulos':
    model = uk.redes.rectangulos(input_size)
elif TIPO_RED=='autoencoder':
    model = uk.redes.autoencoder(input_size)

model.summary()
model.compile(loss="MSE")

# %% [markdown]
# ### Se entrena la red

# %%
history = model.fit(X_train_norm, y_train_norm, epochs=NUM_EPOCAS, validation_data = (X_val_norm, y_val_norm))

# %% [markdown]
# ### Grafica de la loss

# %%
plt.figure()
plt.title("Loss function through training")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(["Training loss", "Validation loss"])

# %% [markdown]
#

# %% [markdown]
# ## Se guardan los pesos y valores de normalizacion

# %%
import os
import json
uk.crear_estructura_carpetas_pesos(RUTA_DATOS_ENTRENAMIENTO)
path = os.path.join(RUTA_DATOS_ENTRENAMIENTO, 'data', TIPO_RED, ELECTRODOMESTICO)

# Guardo los pesos de la red
model.save_weights(os.path.join(path,  'pesos.h5'))

# Guardo los valores de normalizacion de datos
np.save(os.path.join(path,  'std_entrada.npy'), std_entrada)
np.save(os.path.join(path,  'std_salida.npy'), std_salida)

# Guardo la evolucion de la loss de entrenamiento
json.dump(history.history, open(os.path.join(path, 'loss.json'), 'w'))

# %% [markdown]
# ### Se predicen los valores de validación
# Luego de predecir se desnormaliza

# %%
# Prediccion sobre X_val
y_pred_norm = model.predict(X_val_norm)
if TIPO_RED=='autoencoder':
    y_pred_norm = y_pred_norm[:,:,0]

# %%
#Desnormalizo
y_pred = uk.utils.unnormalize_Y(y_pred_norm, std_salida, TIPO_RED)

# %% [markdown]
# ## Grafica de resultados
# Se muestran los 10 primeros valores predecidos

# %%
# Grafico algunos resultados
for i in range(0,10):
    plt.figure()
    if TIPO_RED=='autoencoder':
        y_pred_plot = y_pred[i,:] 
        y_val_plot = y_val[i,:] 
    elif TIPO_RED=='rectangulos':
        y_pred_plot = uk.utils.salida_rectangulos_to_serie_numpy(y_pred[i,:], size=X_val.shape[1])
        y_val_plot = uk.utils.salida_rectangulos_to_serie_numpy(y_val[i,:], size=X_val.shape[1])
    plt.plot(y_pred_plot, label='Predict')
    plt.plot(X_val[i,:,0], label='Input')
    plt.plot(y_val_plot, label='Target')
    plt.grid()
    plt.legend()
