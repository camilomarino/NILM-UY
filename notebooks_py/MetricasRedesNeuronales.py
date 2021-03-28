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
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib notebook
import uk
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Cargo parametros

# %%
ELECTRODOMESTICO = 'kettle' #Electrodomestico para el que se entrenara la red
TIPO_RED = 'rectangulos' #Tipo de red a utilizar
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
x_test_no_visto, y_test_no_visto = uk.cargar_X_y('data', 'test_no_visto', ruta_base = RUTA_DE_DATOS_X_Y)
x_test_visto, y_test_visto = uk.cargar_X_y('data', 'test_visto', ruta_base = RUTA_DE_DATOS_X_Y)

# %%
X_train = x_train[TIPO_RED][ELECTRODOMESTICO]
X_val = x_validacion[TIPO_RED][ELECTRODOMESTICO]
y_train = y_train[TIPO_RED][ELECTRODOMESTICO]
y_val = y_validacion[TIPO_RED][ELECTRODOMESTICO]

X_test_no_visto = x_test_no_visto[TIPO_RED][ELECTRODOMESTICO]
X_test_visto = x_test_visto[TIPO_RED][ELECTRODOMESTICO]
y_test_no_visto = y_test_no_visto[TIPO_RED][ELECTRODOMESTICO]
y_test_visto = y_test_visto[TIPO_RED][ELECTRODOMESTICO]

# El expand dims es necesario para entrenar, tensorflow solicita un vector de 3 dimensiones
X_train = np.expand_dims(
    X_train[:, :-3], axis=2
)
X_val = np.expand_dims(
    X_val[:, :-3], axis=2
)

X_test_no_visto = np.expand_dims(
    X_test_no_visto[:, :-3], axis=2
)
X_test_visto = np.expand_dims(
    X_test_visto[:, :-3], axis=2
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

X_test_no_visto_norm = uk.utils.normalize_X(X_test_no_visto, std_entrada)
X_test_visto_norm = uk.utils.normalize_X(X_test_visto, std_entrada)


if TIPO_RED=='rectangulos':
    std_salida = np.max(y_train[:, 0]) ### Para red de rectangulos
else:
    std_salida = np.max(y_train) ### Para red de autoencoder
y_train_norm = uk.utils.normalize_Y(y_train, std_salida, TIPO_RED)
y_val_norm = uk.utils.normalize_Y(y_val, std_salida, TIPO_RED)


y_test_no_visto_norm = uk.utils.normalize_Y(y_test_no_visto, std_salida, TIPO_RED)
y_test_visto_norm = uk.utils.normalize_Y(y_test_visto, std_salida, TIPO_RED)


# %% [markdown]
# ## Se define el modelo de red neuronal utilizada
# En funcion de los parametros elegidos se carga el modelo adecuado.
#
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
# ### Se cargarn los datos de entrenamiento de la red

# %%
import os
import json
path = os.path.join(RUTA_DATOS_ENTRENAMIENTO, 'data', TIPO_RED, ELECTRODOMESTICO)
model.load_weights(os.path.join(path, 'pesos.h5'))
history = json.load(open(os.path.join(path, 'loss.json'), 'r'))

# %% [markdown]
# ### Grafica de la loss

# %% [markdown]
# Se grafica la "loss". Esto es como evoluciona el error a medida que amuentan las epocas de entrenamiento.

# %%
# %matplotlib inline
plt.figure()
plt.title("Loss function through training")
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(["Training loss", "Validation loss"])


# %% [markdown]
#

# %% [markdown]
# ### Se predicen todos los conjuntos y se compara con los valores reales
# En primera instancia se preciden todos los conjuntos y se grafican en conjunto con la serie real.

# %%
def predict(X):
    y_pred_norm = model.predict(X)
    if TIPO_RED=='autoencoder':
        y_pred_norm = y_pred_norm[:,:,0]
    #Desnormalizo
    y_pred = uk.utils.unnormalize_Y(y_pred_norm, std_salida, TIPO_RED)
    return y_pred

def plot(y_pred, X, y, cantidad=10):
    for i in range(0,cantidad):
        plt.figure()
        if TIPO_RED=='autoencoder':
            y_pred_plot = y_pred[i,:] 
            y_plot = y[i,:] 
        elif TIPO_RED=='rectangulos':
            y_pred_plot = uk.utils.salida_rectangulos_to_serie_numpy(y_pred[i,:], size=X.shape[1])
            y_plot = uk.utils.salida_rectangulos_to_serie_numpy(y[i,:], size=X.shape[1])
        plt.plot(y_pred_plot, label='Predict')
        plt.plot(X[i,:,0], label='Input')
        plt.plot(y_plot, label='Target')
        plt.grid()
        plt.legend()
    


# %% [markdown]
# ## Train

# %%
y_train_pred = predict(X_train_norm)
plot(y_train_pred, X_train, y_train)

# %% [markdown]
# ## Validacion

# %%
y_val_pred = predict(X_val_norm)
plot(y_val_pred, X_val, y_val)

# %% [markdown]
# ## Test no visto

# %%
y_test_no_visto_pred = predict(X_test_no_visto_norm)
plot(y_test_no_visto_pred, X_test_no_visto, y_test_no_visto)

# %% [markdown]
# ## Test visto

# %%
y_test_visto_pred = predict(X_test_visto_norm)
plot(y_test_visto_pred, X_test_visto, y_test_visto)


# %% [markdown]
# # Metricas
# Se reportan varias metricas. Todas estas se pueden ver en [Proyecto de fin de carrera (Marchsesoni-MariÃ±o-Masquil)](https://gitlab.fing.edu.uy/cmarino/base_de_datos_nilm/-/blob/master/docs/MMM20.pdf).
#
# Se dividen en 2 grupos, metricas de regresion y metricas de clasificacion.
#

# %% [markdown]
# #### Metricas de regresion
# \begin{equation}R E I T E=\frac{|\hat{E}-E|}{\max (E, \hat{E})}\end{equation}
# \begin{equation}M A E=\frac{1}{T} \sum_{t=1}^{T}\left|\hat{y}_{t}-y_{t}\right|\end{equation}

# %% [markdown]
# #### Metricas de clasificacion
# \begin{equation}\text { Recall (or True Positive Rate)} = \frac{{TP}}{{TP}+{FN}}\end{equation}
#
# \begin{equation}\text { Precision } =\frac{T P}{T P+F P}\end{equation}
#
# \begin{equation}\text { Accuracy }=\frac{T P+T N}{T P+T N+F P+F N}\end{equation}
#
# \begin{equation}\text { False Positive Rate }=\frac{F P}{T N+F P}\end{equation}
#
#
# \begin{equation}\text { F1-Score }=\frac{2 T P}{2 T P + F P + FN}\end{equation}
#
#
#
#

# %% [markdown]
# #### Curva ROC y AUC
# Es el la grafica de Recall en funcion de False Positive Rate a medida que varia el umbral de clasificacion.
#
# La **AUC** es el area debajo de esta curva, cuanto mas cercana a 1 indica un mayor desempeño.

# %%
def show_results(recalls, precisions, accuracys, fprs, f1s, threshs):
    argmax = np.nanargmax(f1s)
    uk.metricas.plot_roc(recalls, fprs, argmax=argmax)

    reite = uk.metricas.REITE(y_train, y_train_pred, TIPO_RED)
    mae = uk.metricas.MAE(y_train, y_train_pred, TIPO_RED)

    print(f"AUC\t\t\t: {uk.metricas.auc(recalls, fprs)}")
    print("--------------------"*3)
    print("\n")
    print(f"Recall\t\t\t: {recalls[argmax]}")
    print("--------------------"*3)
    print(f"Precision\t\t: {precisions[argmax]}")
    print("--------------------"*3)
    print(f"Accuracy\t\t: {accuracys[argmax]}")
    print("--------------------"*3)
    print(f"False positive rate\t: {fprs[argmax]}")
    print("--------------------"*3)
    print(f"f1-score\t\t: {f1s[argmax]}")
    print("--------------------"*3)
    print("\n")
    print(f"REITE\t\t\t: {reite}")
    print("--------------------"*3)
    print(f"MAE\t\t\t: {mae}")


# %% [markdown]
# ### Train

# %%
recalls, precisions, accuracys, fprs, f1s, threshs = uk.metricas.roc(y_train, y_train_pred, TIPO_RED)
show_results(recalls, precisions, accuracys, fprs, f1s, threshs)

# %% [markdown]
# ### Validación

# %%
recalls, precisions, accuracys, fprs, f1s, threshs = uk.metricas.roc(y_val, y_val_pred, TIPO_RED)
show_results(recalls, precisions, accuracys, fprs, f1s, threshs)

# %% [markdown]
# ### Test no visto

# %%
recalls, precisions, accuracys, fprs, f1s, threshs = uk.metricas.roc(y_test_no_visto, y_test_no_visto_pred, TIPO_RED)
show_results(recalls, precisions, accuracys, fprs, f1s, threshs)

# %% [markdown]
# ### Test Visto

# %%
recalls, precisions, accuracys, fprs, f1s, threshs = uk.metricas.roc(y_test_visto, y_test_visto_pred, TIPO_RED)
show_results(recalls, precisions, accuracys, fprs, f1s, threshs)
