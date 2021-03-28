#!/usr/bin/env python
# coding: utf-8
# %%

# ## Importo librerias

# %%


import os
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
import uk


# ## Cargo datos
# Los datos pueden ser generados mediante el notebook "Procesamiento de datos" o bajarlos desde: https://iie.fing.edu.uy/~cmarino/NILM/vectores.zip
# 
# Nota: no se garantiza que los datos anteriores se encuentren accesibles en un futuro cercano.

# %%


x_train, y_train = uk.cargar_X_y('data', 'train', ruta_base = 'vectores')
x_validacion, y_validacion = uk.cargar_X_y('data', 'validacion', ruta_base = 'vectores')
# x_test_no_visto, y_test_no_visto = uk.cargar_X_y('data', 'test_no_visto', ruta_base = 'vectores')
# x_test_visto, y_test_visto = uk.cargar_X_y('data', 'test_visto', ruta_base = 'vectores')


# #### Se selecciona el electrodomestico y red a utilizar

# %%


for electrodomestico in ['kettle', 'fridge', 'microwave', 'dish']:#'washing'
    for tipo_red in ['rectangulos', 'autoencoder']:
        print(f'''{electrodomestico}, {tipo_red}''')
        X_train = x_train[tipo_red][electrodomestico]
        X_val = x_validacion[tipo_red][electrodomestico]
        X_train = x_train[tipo_red][electrodomestico]
        X_val = x_validacion[tipo_red][electrodomestico]
        Y_train = y_train[tipo_red][electrodomestico]
        Y_val = y_validacion[tipo_red][electrodomestico]

        # El expand dims es necesario para entrenar, tensorflow solicita un vector de 3 dimensiones
        X_train = np.expand_dims(
            X_train[:, :-3], axis=2
        )
        X_val = np.expand_dims(
            X_val[:, :-3], axis=2
        )
        # Normalizacion de X e y
        std_entrada = np.nanmean(np.nanstd(X_train, axis=1))
        X_train_norm = uk.utils.normalize_X(X_train, std_entrada)
        X_val_norm = uk.utils.normalize_X(X_val, std_entrada)

        if tipo_red=='rectangulos':
            std_salida = np.max(Y_train[:, 0]) ### Para red de rectangulos
        else:
            std_salida = np.max(Y_train) ### Para red de autoencoder
        Y_train_norm = uk.utils.normalize_Y(Y_train, std_salida, tipo_red)
        Y_val_norm = uk.utils.normalize_Y(Y_val, std_salida, tipo_red)

        input_size = X_train.shape[1]

        if tipo_red=='rectangulos':
            model = uk.redes.rectangulos(input_size)
        else:
            model = uk.redes.autoencoder(input_size)

        model.summary()
        model.compile(loss="MSE")

        history = model.fit(X_train_norm, Y_train_norm, epochs=50, validation_data = (X_val_norm, Y_val_norm))
        
        uk.crear_estructura_carpetas_pesos('pesos')
        path = os.path.join('pesos', 'data', tipo_red, electrodomestico)
        model.save_weights(os.path.join(path,  'pesos.h5'))
        json.dump(history.history, open(os.path.join(path, 'loss.json'), 'w'))
        del model

