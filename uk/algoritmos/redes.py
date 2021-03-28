# -*- coding: utf-8 -*-

"""Las redes a usar para NILM"""
import os

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

from . import utils


def rectangulos(input_size):
    """Crea la red de rectangulos de baja frec"""
    model = models.Sequential()
    model.add(layers.Conv1D(16, 4, padding="valid", input_shape=(input_size, 1)))
    model.add(layers.Conv1D(16, 4, padding="valid"))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dense(3072, activation="relu"))
    model.add(layers.Dense(2048, activation="relu"))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(3))
    return model


def autoencoder(input_size):
    """Crea el auto encoder de baja frec"""
    hidden = (input_size - 3) * 8
    model = models.Sequential()
    model.add(layers.Conv1D(8, 4, padding="valid", input_shape=(input_size, 1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(hidden, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(hidden, activation="relu"))
    model.add(layers.Reshape(((input_size - 3), 8)))
    model.add(layers.ZeroPadding1D((3, 0)))
    model.add(layers.Conv1D(1, 4, padding="causal"))
    return model


def autoencoder_big(input_size):
    """Variante big del autoencoder"""
    hidden = (input_size - 3) * 8
    model = models.Sequential()
    model.add(layers.Conv1D(8, 4, padding="valid", input_shape=(input_size, 1)))
    model.add(layers.Conv1D(8, 4, padding="valid"))
    model.add(layers.Flatten())
    model.add(layers.Dense(hidden, activation="relu"))
    model.add(layers.Dense(hidden // 4, activation="relu"))
    model.add(layers.Dense(input_size // 10, activation="relu"))
    model.add(layers.Dense(hidden // 4, activation="relu"))
    model.add(layers.Dense(hidden, activation="relu"))
    model.add(layers.Reshape(((input_size - 3), 8)))
    model.add(layers.ZeroPadding1D((3, 0)))
    model.add(layers.Conv1D(1, 4, padding="causal"))
    return model

def rectangulos_hf(input_size):
    """Crea la red de rectangulos de alta frec"""
    model = models.Sequential()
    model.add(layers.Conv1D(16, 4, padding="valid", input_shape=(input_size, 3)))
    model.add(layers.Conv1D(16, 4, padding="valid"))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dense(3072, activation="relu"))
    model.add(layers.Dense(2048, activation="relu"))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(3))
    return model


def autoencoder_hf(input_size):
    """Crea el autoencoder de alta frec"""
    hidden = (input_size - 3) * 8
    model = models.Sequential()
    model.add(layers.Conv1D(8, 4, padding="valid", input_shape=(input_size, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(hidden, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(hidden, activation="relu"))
    model.add(layers.Reshape(((input_size - 3), 8)))
    model.add(layers.ZeroPadding1D((3, 0)))
    model.add(layers.Conv1D(1, 4, padding="causal"))
    return model


def make_predictions(modelname, freq, dataset, input_size, X, datadir):
    """Hace predicciones con un modelo pre entrenado"""
    # Decision de modelo
    if modelname == "autoencoder_big":
        model = autoencoder_big(input_size)
    elif modelname == "rectangulos" and freq == "hf":
        model = rectangulos_hf(input_size)
    elif modelname == "rectangulos" and freq == "lf":
        model = rectangulos(input_size)
    elif modelname == "autoencoder" and freq == "hf":
        model = autoencoder_hf(input_size)
    else:
        model = autoencoder(input_size)
    model.load_weights(os.path.join(datadir, "checkpoint_callback.h5"))
    # Cargo datos de entrenamiento
    if freq == "hf":
        X = X[:,:,1:]
        # Relleno nans con 0
        X = np.nan_to_num(X)
        # Borro ventanas enteras de 0
        to_keep = np.sum(np.sum(X, axis=1), axis=1) != 0
        X = X[to_keep]
        # Normalizacion de X en hf
        std = np.nanstd(X, axis=1)
        std = np.nanmean(std, axis=0)
        X = (X - np.nanmean(X, axis=1, keepdims=True)) / std
    else:
        # El expand dims es necesario para la lf
        X = np.expand_dims(X, axis=2)
        # Normalizacion de X en lf
        std = np.std(X, axis=1)
        std = np.mean(std)
        X = utils.normalize_X(X, std)
    y = model.predict(X)
    np.save(os.path.join(datadir, "predictions" + "_" + dataset), y)