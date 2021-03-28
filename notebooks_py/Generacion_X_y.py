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
# ### Celda Colab
# Si se corre en Colab tendra que montar su google drive y seleccionar en ***PATH_DRIVE*** el directorio en el que esta clonado su repositorio

# %%
PATH_DRIVE =  '/content/drive/My Drive/base_de_datos_nilm'
import os
if 'COLAB_GPU' in os.environ:
   print("Estoy corriendo en Colab")
   from google.colab import drive
   drive.mount('/content/drive')
   %cd $PATH_DRIVE
   
else:
   print("NO estoy corriendo en Colab")

# %% [markdown]
# ## Importo Librerias

# %%
import copy
import uk
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Carga de datos
# Se tienen datos de 5 casas de Reino Unido (UK). La carga consiste en adquirir el .h5 que suministra el autor y cargalo como dataframe (objecto pandas).
# Una vez cargados, se los divide en 2 tramos, los datos de inicio y los datos de fin.
#
# **datos_ini/fin** : diccionarios cuya primera key es el numero de casa y su segunda key es el nombre del electrodomestico.
#
# Nota: el .h5 se puede encontrar en: 
#
# https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated

# %%
ruta = 'ukdale.h5'
datos = uk.carga_datos_desde_hdf(ruta, verbose=True)
datos_ini, datos_fin = uk.separacion_datos(datos, dias_final=14, verbose=True)
del datos #libero ram ya que ocupa al rededor de 7gb de RAM

# %% [markdown]
# #### Guardar procesamiento
# Dado que el procesamiento anterior es costoso, se sumintran funciones para guardar y cargar los datos procesados.
# Una vez calculados, sientase libre de guardar los datos y no procesarlos más.
#
# Si el procesamiento fuera muy costoso, se puede descargar los datos ya procesados desde: 
#
# https://iie.fing.edu.uy/~cmarino/NILM/datos_ini.pickle
#
# https://iie.fing.edu.uy/~cmarino/NILM/datos_fin.pickle
#
# Nota: no se garantiza que los datos anteriores se encuentren accesibles en un futuro cercano.
#

# %%
# # La siguiente celda sirve para guardar los datos
# uk.guardar(datos_ini, 'datos_ini.pickle')
# uk.guardar(datos_fin, 'datos_fin.pickle')

# %%
# # La siguiente celda sirve para cargar los datos
# datos_ini = uk.cargar('datos_ini.pickle')
# datos_fin = uk.cargar('datos_fin.pickle')

# %% [markdown]
# ## Activaciones
# Una activación de un electrodomestico corresponde a un ciclo de trabajo del mismo.
# <br><br>
# ### Parametros de calculo de activaciones
# *Estos parametros son utilizados para determinar en la serie desagregada de consumo cuando se encuentra una activación. Estos aparecen en **"Table 4: Arguments passed to get_activations()." del paper de Jack Kelly**.*
# <br><br>
# Descripción de parametros:
#
# **on_power_threshold** : Umbral en Watts que determina si un electrodomestico esta encendido o no
#
# **min_on_duration**    : Duración mínima en minutos que debe tener una activación (largo mínimo del ciclo de trabajo)
#
# **min_off_duration**   : Duración mínima en minutos que debe estar apagado(por debajo de "on_power_threshold") el electrodomestico para que se considere que finalizo la activación.
# <br><br>
# Nota: Jack Kelly en su publicacion utiliza "Max power", pero acá no es utilizado, esto se debe a que la propia función que él suministra NILM-TK no la utiliza.

# %%
# Definicion de parametros
# Parámetros de las activaciones
parametros_kettle = {"min_off_duration": 0, "min_on_duration": 12, 
                     "on_power_threshold": 2000}
parametros_fridge = {"min_off_duration": 12, "min_on_duration": 60, 
                     "on_power_threshold": 50}
parametros_washing = {"min_off_duration": 160, "min_on_duration": 1800, 
                      "on_power_threshold": 20}
parametros_microwave = {"min_off_duration": 30, "min_on_duration": 12, 
                        "on_power_threshold": 200}
parametros_dish = {"min_off_duration": 1800, "min_on_duration": 1800, 
                   "on_power_threshold": 10}

parametros = {"kettle": parametros_kettle, "fridge": parametros_fridge, 
              "washing": parametros_washing, "microwave": parametros_microwave,
              "dish": parametros_dish }





# %% [markdown]
# ## Ventanas de activaciones
# Las ventanas son utilizadas para el entrenamiento de las redes neuronales profundas utilizadas. Los vectores de entrenamiento X serán un conjunto de ventanas (las filas de la matriz X son una ventana).
# Una ventana es un tramo de serie temporal que contiene o no una activación. Se calcularan ventanas con activaciones y ventanas sin activaciones para cada uno de los electrodomésticos. 
# <br><br>
# ### Parametros de largo de ventanas
# *Estos parametros son los que aparecen en la sección **"3. TRAINING DATA"**, en particualar dice "The window width is decided on an appliance-by-appliance basis and varies from 128 samples (13 minutes) for the kettle to 1536 samples (2.5 hours) for the dish washer."*
# <br><br>
# El parametro fundamental para generar una ventana es su largo o "win_len".
# **win_len** : Es la duración en minutos de la ventana.
# <br>
# Nota: en el articulo de referencia no se explicita el largo para todos los electrodomésticos, por lo tanto se eligen por inspección sobre train.

# %%
# Largos de ventana
win_len = {"kettle": 13, "fridge": 60, "washing": 180, "microwave": 10,
           "dish": 150 }

# %% [markdown]
# ## Semilla
# Se fija la semilla para el generador de numeros pseudoaleatorio.

# %%
seed = 5

# %% [markdown]
# ## Calculo de vectores de UN solo electrodomestico
# Se muestra modo de ejemplo como se utiliza la funcion crear_vectores(...).
#
# Esta recibe los siguientes parametros:<br>
# **datos**            : conjunto de datos a utilizar<br>
# **electrodomestico** : nombre del electrodomestico para el cual se calular los vectores<br>
# **numeros_casas**    : lista de numeros de casas que se usan para calcular los vectores<br>
# **parametros**       : los parametros de calculo de activaciones<br>
# **largo_de_ventana** : el largo de ventana en minutos<br>
# **tipo_red**         : pueden generarse datos para una arquitectura de **RECTANGULOS O AUTOENCODER**<br>
# **seed**             : el numero de semilla<br>
# **verbose**          : modo verboso

# %%
# #Ejemplo de creacion de vectores de un electrodomestico para una unica casa
x, y = uk.crear_vectores(datos = datos_ini, 
                                electrodomestico = 'fridge', 
                                numeros_casas = [2],
                                parametros = parametros['fridge'], 
                                largo_de_ventana = win_len['fridge'],
                                tipo_red = 'autoencoder', 
                                seed = seed,                            
                                verbose = True)


# %% [markdown]
# ## Calculo de vectores de TODOS los electrodomesticos
# Se suministra un script capaz de calcular todas las activaciones en una estructura de diccionario.
#
# Tambien se implementaron una serie de funciones para el guardado automatico de vectores con este metodo.

# %% [markdown]
# ### Se determinan las casas a utilizar para cada electrodomestico
# Estos parametros determinan que casas se utilizar para train y test

# %%
# Ejemplo de como es posible crear todos los x,y en loop
# La forma de los vectores que contendran los datos es la  siguiente:
# x_test/train_visto/novisto['tipo_red']['electrodomestico']
electrodomesticos = ['kettle', 'fridge', 'washing', 'microwave', 'dish']
numeros_casas_train_validacion = {'kettle' : [1,2,3,4], 
                                   'fridge' : [1,2,4], 
                                   'washing' : [1,5], 
                                   'microwave' : [1,2], 
                                   'dish' : [1,2],
                                   }
numeros_casas_test_no_visto = {'kettle' : [5], 
                               'fridge' : [5], 
                               'washing' : [2], 
                               'microwave' : [5], 
                               'dish' : [5],
                               }


# %% [markdown]
# ### Se calculan los vectores
# Los vectores de numpy se encuentran agrupados en dobles diccionarios.<br>
#
# Por ejemplo, x_train_validacion['autoencoder']['kettle'] tendra el vector X de tipo autoencoder para la heladera.<br>
#
# Se tiene una estructura de doble diccionario para cada conjunto (test, test no visto, train y validacion).
#

# %%
tipos_red = ['autoencoder', 'rectangulos']


modelo_vector = {'autoencoder' : {'kettle' : None,
                                        'fridge' : None,
                                        'washing' : None,
                                        'microwave' : None,
                                        'dish' : None},
                'rectangulos' : {'kettle' : None,
                                  'fridge' : None,
                                  'washing' : None,
                                  'microwave' : None,
                                  'dish' : None}
                }

# %%

# Se copia la estructura de doble diccionario (dado que es diccionario doble se usa deepcopy)
x_train_validacion = copy.deepcopy(modelo_vector)
y_train_validacion = copy.deepcopy(modelo_vector)
x_train = copy.deepcopy(modelo_vector)
y_train = copy.deepcopy(modelo_vector)
x_validacion = copy.deepcopy(modelo_vector)
y_validacion = copy.deepcopy(modelo_vector)
x_test_no_visto = copy.deepcopy(modelo_vector)
y_test_no_visto = copy.deepcopy(modelo_vector)
x_test_visto = copy.deepcopy(modelo_vector)
y_test_visto = copy.deepcopy(modelo_vector)

#Calculo de datos de train-validacion
for tipo_red in tipos_red:
    for elec in electrodomesticos:
        x_train_validacion[tipo_red][elec], y_train_validacion[tipo_red][elec] = (
                uk.crear_vectores(
                        datos = datos_ini, 
                        electrodomestico = elec, 
                        numeros_casas = numeros_casas_train_validacion[elec],
                        parametros = parametros[elec], 
                        largo_de_ventana = win_len[elec],
                        tipo_red = tipo_red, 
                        seed = seed,
                        verbose = True)
                )
        
#Se divide en conjunto de test y validacion
from sklearn.model_selection import train_test_split
val_size = 0.2
for tipo_red in tipos_red:
    for elec in electrodomesticos:
        (x_train[tipo_red][elec], x_validacion[tipo_red][elec], 
        y_train[tipo_red][elec], y_validacion[tipo_red][elec]) = (
                train_test_split(x_train_validacion[tipo_red][elec], 
                                 y_train_validacion[tipo_red][elec], 
                                 test_size = val_size,
                                 random_state = seed)
                )
del x_train_validacion, y_train_validacion

#Calculo de datos de test no visto
for tipo_red in tipos_red:
    for elec in electrodomesticos:
        x_test_no_visto[tipo_red][elec], y_test_no_visto[tipo_red][elec] = (
                uk.crear_vectores(
                        datos = datos_ini, 
                        electrodomestico = elec, 
                        numeros_casas = numeros_casas_test_no_visto[elec],
                        parametros = parametros[elec], 
                        largo_de_ventana = win_len[elec],
                        tipo_red = tipo_red, 
                        seed = seed,
                        verbose = True)
                )

#calculo de datos de test visto
for tipo_red in tipos_red:
    for elec in electrodomesticos:
        x_test_visto[tipo_red][elec], y_test_visto[tipo_red][elec] = (
                uk.crear_vectores(
                        datos = datos_fin, 
                        electrodomestico = elec, 
                        numeros_casas = numeros_casas_train_validacion[elec],
                        parametros = parametros[elec], 
                        largo_de_ventana = win_len[elec],
                        tipo_red = tipo_red, 
                        seed = seed,
                        verbose = True)
                )

# %% [markdown]
# ### Generacion de datos sinteticos

# %%
numeros_casas_sintetico = {'kettle' : [1,2,3,4], 
                           'fridge' : [1,2,4], 
                           'washing' : [1,5], 
                           'microwave' : [1,2], 
                           'dish' : [1,2],
                           }


# %%
x_sintetico = copy.deepcopy(modelo_vector)
y_sintetico = copy.deepcopy(modelo_vector)

# %%
for tipo_red in tipos_red:
    for elec in electrodomesticos:
        x_sintetico[tipo_red][elec], y_sintetico[tipo_red][elec] = (
                uk.crear_vectores_sinteticos(
                        datos = datos_ini, 
                        electrodomestico_base = elec, 
                        electrodemsticos_distractores = [e for e in electrodomesticos if e!=elec],
                        numeros_casas = numeros_casas_sintetico[elec],
                        parametros = parametros, 
                        largo_de_ventana = win_len[elec],
                        tipo_red = tipo_red, 
                        cant_a_calcular = 1000,
                        prob_distractores = 0.5, 
                        seed = seed)
                )

# %% [markdown]
# ### Guardar procesamiento
# Dada la estrucutra propuesta, se suministran funciones para el guardado automatico de estos vectores. <br>
# Se recomienda, para cada conjunto de parametros utlizados, guardar los vectores y no recalular cada vez.
#
# La ruza base indica el directorio raiz de el arbol de archivos.
#
# guardar_X_y(...) recibe:<br>
# **x** : el vector X (nummpy array)<br>
# **y** : el vector objetivo Y (nummpy array)<br>
# **tipo_data** : el nombre de los datos (brinda la posibilidad de integrarlo con datos de alta frecuencia en la misma estructira de carpetas)<br>
# **ruta_base** : directorio raiz

# %%
#Ejemplo de como guardar los datos
## Genera la estructura de carpets necesaria
uk.crear_estructura_carpetas(ruta_base = 'vectores')
## Guarda los datos X e y como .npy
uk.guardar_X_y(x_train, y_train, 'data', 'train', ruta_base = 'vectores')
uk.guardar_X_y(x_validacion, y_validacion, 'data', 'validacion', ruta_base = 'vectores')
uk.guardar_X_y(x_test_no_visto, y_test_no_visto, 'data', 'test_no_visto', ruta_base = 'vectores')
uk.guardar_X_y(x_test_visto, y_test_visto, 'data', 'test_visto', ruta_base = 'vectores')
uk.guardar_X_y(x_sintetico, y_sintetico, 'data', 'sintetico', ruta_base = 'vectores')

# %% [markdown]
# ### Cargar procesamiento
# En caso que el procesamiento sea costoso, se pueden descargar los datos procesados (con los parametros mostrados en este notebook) desde:
#
# https://iie.fing.edu.uy/~cmarino/NILM/vectores.zip
#
# Nota: no se garantiza que los datos anteriores se encuentren accesibles en un futuro cercano.

# %%
# #Ejemplo de como cargar datos guardados
# x_train, y_train = uk.cargar_X_y('data', 'train', ruta_base = 'vectores')
# x_validacion, y_validacion = uk.cargar_X_y('data', 'validacion', ruta_base = 'vectores')
# x_test_no_visto, y_test_no_visto = uk.cargar_X_y('data', 'test_no_visto', ruta_base = 'vectores')
# x_test_visto, y_test_visto = uk.cargar_X_y('data', 'test_visto', ruta_base = 'vectores')

# %% [markdown]
# ### Plot de algunos ejemplos
# Una forma de visualizar los datos es imprimiendo directamente las filas de los X e Y.
#
# Utilizando la potencialidad de que las ultimas 3 columnas de X guardamos metadatos (timestamp de inicio, timestamp de fin y numero de casa) es posible recuperar la serie de pandas agregada y desagregada y hacer un plot con estos datos.
#
# La funcion visualizar(...) recibe: <br>
# **datos** : el doble diccionario de datos<br>
# **x** : el vector X<br>
# **y** : el vector Y<br>
# **elec** : el nombre de electrodomestico de los datos

# %%
#Visualizacion de datos en forma de dataframe
# %matplotlib notebook
elec = 'dish'
i = 2
uk.visualizar(datos_ini, x_train['autoencoder'][elec][i], 
                      y_train['autoencoder'][elec][i], elec)
uk.visualizar(datos_ini, x_train['rectangulos'][elec][i], 
                      y_train['rectangulos'][elec][i], elec)
