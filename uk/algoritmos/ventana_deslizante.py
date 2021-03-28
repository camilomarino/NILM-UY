# -*- coding: utf-8 -*-

# la documentacion de este modulo es muy pobre
# Esto esta harcodeado para que funcione con largos de 

import numpy as np
from tqdm.notebook import tqdm


from . import redes as nets

# Diccionario de valores de largo de ventana por defecto
input_size_default = {
    "kettle": 130,
    "dish": 1500,
    "washing": 1800,
    "fridge": 600,
    "microwave": 100,
}


def get_Y_ventana_deslizante(y_pred, y_real, input_size):
    """Toma dos vectores de y (predict y desagregada real) y devuelve la 
    matriz Y que son los y por ventana"""
    n_windows = np.ceil(y_pred.size / input_size).astype(int)
    pred = np.zeros([n_windows * input_size])
    real = np.zeros([n_windows * input_size])
    pred[: y_pred.size] = y_pred.copy()
    real[: y_real.size] = y_real.copy()
    return (
        np.reshape(pred, [n_windows, input_size]),
        np.reshape(real, [n_windows, input_size]),
    )


def get_X_Y_ventana_deslizante(serie, elec, largo_de_ventana=130, stride=1, freq="lf"):
    if freq == "hf":
        try:
            serie_aggregate = serie["aggregate"].to_numpy()[:, 0]
        except:
            serie_aggregate = serie["aggregate"].to_numpy()
        serie_aggregate_form_factor = serie["form factor"].to_numpy()
        serie_aggregate_phase = serie["phase"].to_numpy()
        serie_aggregate_power = serie["power"].to_numpy()
        serie_elec = serie[elec].to_numpy()[:, 0]

        X = np.zeros([int(len(serie_aggregate) / stride), largo_de_ventana, 3])
        Y = np.zeros([int(len(serie_aggregate) / stride), largo_de_ventana])
        i = 0
        j = 0
        while i < len(serie_aggregate) - largo_de_ventana:
            X[j, :, 0] = serie_aggregate_form_factor[i : i + largo_de_ventana]
            X[j, :, 1] = serie_aggregate_phase[i : i + largo_de_ventana]
            X[j, :, 2] = serie_aggregate_power[i : i + largo_de_ventana]
            Y[j] = serie_elec[i : i + largo_de_ventana]
            i += stride
            j += 1

        return X, Y

    elif freq == "lf":
        try:
            serie_aggregate = serie["aggregate"].to_numpy()[:, 0]
        except:
            serie_aggregate = serie["aggregate"].to_numpy()
        serie_elec = serie[elec].to_numpy()[:, 0]

        X = np.zeros([int(len(serie_aggregate) / stride), largo_de_ventana])
        Y = np.zeros([int(len(serie_aggregate) / stride), largo_de_ventana])
        i = 0
        j = 0
        while i < len(serie_aggregate) - largo_de_ventana:
            X[j] = serie_aggregate[i : i + largo_de_ventana]
            Y[j] = serie_elec[i : i + largo_de_ventana]
            i += stride
            j += 1

        return X, Y

def overlap_predictions(
    serie,
    std_entrada,
    std_salida,
    electrodomestico,
    path_weights,
    input_size,
    modelname="rectangulos",
    freq="lf",
):
    # Cargo el modelo de la red
    # input_size = input_size_dict[electrodomestico]
    # Decision de modelo
    if modelname == "autoencoder_big":
        model = nets.autoencoder_big(input_size)
    elif modelname == "rectangulos" and freq == "hf":
        model = nets.rectangulos_hf(input_size)
    elif modelname == "rectangulos" and freq == "lf":
        model = nets.rectangulos(input_size)
    elif modelname == "autoencoder" and freq == "hf":
        model = nets.autoencoder_hf(input_size)
    else:
        model = nets.autoencoder(input_size)
    if modelname == "autoencoder_big":
        modelname = "autoencoder"
    model.load_weights(path_weights)

    # Creo el X overlap para predecir por la red
    X_overlap, _ = get_X_Y_ventana_deslizante(
        serie, electrodomestico, largo_de_ventana=input_size, stride=1, freq=freq
    )

    # Borro nan
    X_overlap = np.nan_to_num(X_overlap)

    # Normalizacion
    if freq == "lf":
        # Expando dimensiones para poder entrar a la red
        X_overlap = np.expand_dims(X_overlap, axis=2)
        # Normalizo
        media = np.mean(X_overlap, axis=1, keepdims=True)
        X_overlap = (X_overlap - media) / std_entrada

    elif freq == "hf":
        X_overlap = (
            X_overlap - np.nanmean(X_overlap, axis=1, keepdims=True)
        ) / std_entrada

    # Predict por la red
    y_overlap = model.predict(X_overlap, verbose=1)

    if modelname == "autoencoder":
        y_overlap = y_overlap[:, :, 0]

    if modelname == "rectangulos":
        y_overlap_grande = np.empty_like(X_overlap[:, :, 0])
    # Borro para ahorra memoria
    del X_overlap
    if modelname == "rectangulos":
        for i in range(y_overlap.shape[0]):

            extremo_inicial_pred = max(
                int(y_overlap[i, 1] * y_overlap_grande.shape[1]), 0
            )
            extremo_final_pred = min(
                int(y_overlap[i, 2] * y_overlap_grande.shape[1]),
                y_overlap_grande.shape[1] - 1,
            )

            y_overlap_grande[i, extremo_inicial_pred:extremo_final_pred] = y_overlap[
                i, 0
            ]
        y_overlap = y_overlap_grande

    # Comienza el promediado de series superpuestas
    y_prom = np.zeros((len(serie["aggregate"])))

    # PRINCIO
    for i in range(input_size - 1):
        for j in range(0, i + 1):
            y_prom[i] += y_overlap[j, i - j]

        y_prom[i] /= j + 1

    # MEDIO
    y_prom[-(input_size):] = y_overlap[-1]
    for muestra in tqdm(range(input_size - 1, len(serie["aggregate"]) - input_size)):
        fila_base = muestra - (input_size - 1)
        for j in range(input_size):

            y_prom[muestra] += y_overlap[fila_base + j, input_size - 1 - j]

        y_prom[muestra] /= input_size

    y_prom *= std_salida

    # Calibracion por promedio
    calibracion = {"kettle": 1.5093945550126446, 
                   "dish": 2.100864805054888, 
                   "washing": 13.288613619594932, 
                   "fridge": 3.148107780439606, 
                   "microwave": 1.6091764317544148}
    y_prom *= calibracion[electrodomestico]
    return y_prom


def cut_serie(serie, house, elec, freq, dia_inicio=0, dias_total=10):
    """Dado un diccionario(serie) con inidice de numero de casas, devuelve
    un diccionario con inid"""

    casa = serie[house]
    serie_ = serie[house].copy()
    
    
    where_elec , _ = np.where(serie_[elec] != 0)
    if freq=="lf":
        where_agg = np.where(serie_["aggregate"] != 0)[0]
    elif freq=="hf":
        where_agg= np.where(serie_["power"] != 0)[0]
    
    inicio = max(where_agg[0],where_elec[0]) + 10*60*24*dia_inicio
    fin = min(inicio + 10*60*24*dias_total , where_agg[-1],where_elec[-1])
    
    largo = fin-inicio
    print("------------------------------",largo, "------------------------------")
    
    elecs = list(casa.keys())
    elecs.remove("num_casa")
    for elec_ in elecs:
        
        #Recalibracion de la casa 5
        if freq=="hf" and house==5 and elec_=="power":
            serie_[elec_] = casa[elec_][inicio:fin]*2.3988152075321976
        else:
            serie_[elec_] = casa[elec_][inicio:fin]

    # if house==6:
    #     serie_["aggregate"] = casa["power"][inicio:fin]
	
    elecs.remove(elec)
    elecs.remove("aggregate")
    if freq=="hf":
        elecs.remove("power")
        elecs.remove("form factor")
        elecs.remove("phase")
    
    for elec_ in elecs:
        del serie_[elec_]
    
    

    return serie_


#std_entrada, std_salida, pesos(PATH), casa(datos_ini/fin)
def generar_pred(*,std_entrada: np.ndarray,
                 std_salida: np.ndarray,
                 datos: dict,
                 ruta_pesos: str,
                 electrodomestico: str,
                 num_casa: int, 
                 modelname: str,
                 input_size: int,
                 rango_dias: int = 20):
    """Funcion para generar predicciones de ventana deslizante"""
    
    if int(num_casa) not in datos:
        raise AttributeError('El numero de casa seleccionado no es correcto '\
                             'para estos datos')
    if electrodomestico not in datos[int(num_casa)]:
        raise AttributeError('El electrodomestico seleccionado no es correcto '\
                             'para estos datos')
        
    
    # Cargo datos de serie agregada  
    
    y_prom_predict = np.array(())
    y_prom_real = np.array(())
    
        
    agg = np.array(())
         

    # import ipdb
    # ipdb.set_trace()
    where_elec , _ = np.where(datos[int(num_casa)][electrodomestico] != 0)
    where_agg = np.where(datos[int(num_casa)]["aggregate"] != 0)[0]
    
    inicio = max(where_agg[0],where_elec[0])
    fin = min(where_agg[-1],where_elec[-1])
    
    largo = (fin-inicio)/(10*60*24)
    
    cantidad = int(np.ceil(largo/rango_dias))
    print(f'Se divide el calculo en {cantidad} tramos de '\
          f'{int(round(min(rango_dias, largo)))} días')   
    print('-'*70, '\n\n')
    for i in (range(cantidad)):
        print(f'Tramo {i}\n')
        paso = rango_dias
        dias_total = rango_dias
        serie = cut_serie(datos, int(num_casa), electrodomestico, 'lf', 
                          i*paso, 
                          dias_total)
        # del casa
    
        y_prom_predict_ = overlap_predictions(
            serie,
            std_entrada,
            std_salida,
            electrodomestico,
            ruta_pesos,
            modelname=modelname,
            input_size=input_size,
            freq='lf',
        )
        
        y_prom_predict = np.hstack((y_prom_predict, y_prom_predict_))
        y_prom_real = np.hstack((y_prom_real, 
                                 serie[electrodomestico].to_numpy()[:,0]))

        try:
            agg = np.hstack((agg, 
                         serie["aggregate"].to_numpy()[:, 0]))
        except:
            agg = np.hstack((agg, 
                         serie["aggregate"].to_numpy()))      
        print('-'*70, '\n\n\n')
 
    return y_prom_predict, y_prom_real, agg