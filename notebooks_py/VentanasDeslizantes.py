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
# # Importar librerias

# %%
import numpy as np
import uk
import matplotlib.pyplot as plt
# %matplotlib notebook
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Fijar parametros

# %%
ELECTRODOMESTICO = 'kettle' #Electrodomestico para el que se entrenara la red
TIPO_RED = 'autoencoder' #Tipo de red a utilizar
RUTA_DE_DATOS_PICKLE = '' #ruta en la que se encuentralos pickles que contienen todas las series temporales
RUTA_DATOS_ENTRENAMIENTO = 'pesos' #Carpeta en la que se guardaran los datos generados en el entrenamiento
DATOS_FIN_o_INICIO = 'fin' # Indica si se van a usar las ultimas 2 semanas(fin) o todo el resto(inicio) 
                               # o datos de uruguayUruguay
NUMERO_CASA = 1 #numero de casa, puede ser [1,2,3,4,5] para los datos de UK o [6,7] para los de UY
INPUT_SIZE = uk.ventana_deslizante.input_size_default[ELECTRODOMESTICO] 

# %% [markdown]
# ## Cargar datos

# %%
from os.path import join
import json
path = join(RUTA_DATOS_ENTRENAMIENTO, 'data', TIPO_RED, ELECTRODOMESTICO)
RUTA_PESOS = join(path, 'pesos.h5')
history = json.load(open(join(path, 'loss.json'), 'r'))
std_entrada = np.load(join(path, 'std_entrada.npy'))
std_salida = np.load(join(path, 'std_salida.npy'))

if DATOS_FIN_o_INICIO=='inicio':
    datos = uk.cargar(join(RUTA_DE_DATOS_PICKLE, 'datos_ini.pickle'))
elif DATOS_FIN_o_INICIO=='fin':
    datos = uk.cargar(join(RUTA_DE_DATOS_PICKLE, 'datos_fin.pickle'))
elif DATOS_FIN_o_INICIO=='uruguay':
    datos = uk.cargar(join(RUTA_DE_DATOS_PICKLE, 'datos_uruguay.pickle'))

# %% [markdown]
# ## Se calcula los vectores por el metodo de ventanas deslizantes

# %%
y_predict, y_real, aggregate = uk.ventana_deslizante.generar_pred(std_entrada = std_entrada,
                                                                     std_salida =  std_salida,
                                                                     datos = datos,
                                                                     ruta_pesos = RUTA_PESOS,
                                                                     electrodomestico = ELECTRODOMESTICO,
                                                                     num_casa = NUMERO_CASA, 
                                                                     modelname = TIPO_RED, 
                                                                     input_size = INPUT_SIZE)

# %% [markdown]
# ## Plot de la ventana predecida

# %%
plt.figure()
plt.plot(y_predict, '-', label='y predict')
plt.plot(y_real, '--', label='y real')
plt.plot(aggregate, '-.',label='aggregate')
plt.legend()

# %% [markdown]
# ## Metricas

# %%
y_ventana_predict, y_ventana_real = uk.ventana_deslizante.get_Y_ventana_deslizante(y_predict, y_real, INPUT_SIZE)

# %%
epsilon = 0.0001
recalls, precisions, accuracys, fprs, f1s, threshs = (
    uk.metricas.roc(y_ventana_real+epsilon, y_ventana_predict, TIPO_RED, True, elec=ELECTRODOMESTICO)
    )


# %%
def show_results(recalls, precisions, accuracys, fprs, f1s, threshs, y_real, y_predict, plot_auc=True):
    argmax = np.nanargmax(f1s)
    umbral = threshs[argmax]
    
    if plot_auc: 
        uk.metricas.plot_roc(recalls, fprs, elec=ELECTRODOMESTICO, argmax=argmax, umbral=umbral)
    
    reite = uk.metricas.REITE(y_real, y_predict, 'ventanas')
    mae = uk.metricas.MAE(y_real, y_predict, 'ventanas')

    if plot_auc:
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
    return argmax


# %%
argmax = show_results(recalls, precisions, accuracys, fprs, f1s, threshs, y_real, y_predict)
umbral = threshs[argmax]


# %% [markdown]
# #  Umbralizacion
# Ahora se realiza una umbralizacion sumamente sencilla. Se "matan" los valores por arriba del umbral (se los ponen en 0), ademas se multiplica por un factor arbitrario 0.8. Por ultimo, se limita la prediccion a el valor de la serie agregada.

# %%
umbral

# %%
y_predict_umbralizada = y_predict.copy()
y_predict_umbralizada[y_predict_umbralizada<umbral] = 0.00001 #Si ponemos 0 se producen errores numericos
y_predict_umbralizada[y_predict_umbralizada>aggregate] = aggregate[y_predict_umbralizada>aggregate]

# %%
plt.figure()
#plt.plot(y_predict, '-', label='y predict')
plt.plot(y_real, '--', label='y real')
plt.plot(aggregate, '-.',label='aggregate')
plt.plot(y_predict_umbralizada, '-', label='y predict umbralizada', alpha=0.7)
plt.legend()

# %%
reite = uk.metricas.REITE(y_real, y_predict_umbralizada, 'ventanas')
mae = uk.metricas.MAE(y_real, y_predict_umbralizada, 'ventanas')
print(f"\nREITE\t\t\t: {reite}")
print("--------------------"*3)
print(f"MAE\t\t\t: {mae}\n")

# %%
print(np.mean(y_real))
print(np.mean(y_predict))
print(np.mean(y_predict_umbralizada))
