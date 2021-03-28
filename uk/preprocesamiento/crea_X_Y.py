import numpy as np
from .crea_X_Y_utils import *
from datetime import timedelta
from sklearn.utils import shuffle


def crear_vectores(datos:dict, electrodomestico:str, numeros_casas:list,
                 parametros:dict, largo_de_ventana:int,
                 tipo_red:str='rectangulos', seed:bool=None,
                 verbose=False):
    '''
    Parameters
    ----------
    datos : dict
        dicccionario caragado via el modulo carga_datos.
    electrodomestico : str
        Nombre de electrodomestico entre los posibles 5.
    numeros_casas : list
        Lista de los numeros de casas. Puede valer 1,2,3,4,5.
    parametros : dict
        Diccionario de los 3 parametros fundamentales.
            "min_off_duration",
            "min_on_duration",
            "on_power_threshold"
    largo_de_ventana : int
        Minutos a utilizar para el largo de ventana
    tipo_red : str, optional
        Tipo de red para el cual se generan datos. Puede ser:
            'rectangulos',
            'autoencoder'
        The default is 'rectangulos'.
    seed: bool
        Numero de semilla para la generacion aleatoria de datos
    verbose: bool
        True activa el modo verboso

    Returns
    -------
    TYPE
        Devuelve una pareja X,Y de matrices de numpy con los vectores 
        correspondientes.
    
    Ejemplo
    -------
        x, y = uk.crear_vectores(datos = datos_ini, 
                                electrodomestico = 'fridge', 
                                numeros_casas = [2],
                                parametros = parametros['fridge'], 
                                largo_de_ventana = win_len['fridge'],
                                tipo_red = 'autoencoder', 
                                seed = seed,                            
                                verbose = True)

    '''

    #Algunos controloes de errores
    if not(isinstance(datos, dict) and isinstance(electrodomestico, str) \
               and isinstance(numeros_casas, list) and isinstance(parametros, dict) \
               and isinstance(largo_de_ventana, int) and isinstance(tipo_red, str)):
        raise TypeError('Tipo de dato incorrecto')
    
    if not set(datos.keys()).issubset({1,2,3,4,5}):
        raise ValueError('Las keys del diccionario de datos deben ir de\
                         1 a 5')
    
    nombres_electrodomestico = {'kettle', 'fridge', 'washing', 'microwave',
                                'dish'}    
    if not {electrodomestico}.issubset(nombres_electrodomestico):
        raise ValueError('El electrodomestico no es valido')
    
    
    for num_casa in numeros_casas:
        if not isinstance(num_casa, int):
            raise TypeError('La lista de numeros_casas debe ser de int')
        elif num_casa>5 or num_casa<1:
            raise ValueError('La lista de numeros_casas deben ser alguno \
                             de los siguientes numeros 1,2,3,4,5')
    
    for num_casa in numeros_casas:
        if not {electrodomestico}.issubset(set(datos[num_casa].keys())):
            raise ValueError(f'La casa {num_casa} no tiene datos para \
                             el electrodomestico {electrodomestico}')
        
                             
                             
    nombre_parametros = {"min_off_duration", "min_on_duration",
                             "on_power_threshold"}
    if set(parametros.keys()) != nombre_parametros:
        raise ValueError('Keys del diccionario de parametro incorrecta')
    
    if largo_de_ventana<1:
        raise ValueError('largo_de_ventana debe ser mayor a 0')
        
    if tipo_red!='rectangulos' and tipo_red!='autoencoder':
        raise ValueError('El tipo_red debe ser rectangulos o autoencoder')
        
                
        
    datos_a_procesar = [datos[i] for i in numeros_casas]
    X_act, Y_act, X_no_act, Y_no_act = (
                matrices_de_activaciones(datos_a_procesar, electrodomestico, 
                                         tipo_red, parametros, 
                                         timedelta(minutes=largo_de_ventana),
                                         seed=seed,
                                         verbose=verbose))
        
    X = np.vstack((X_act, X_no_act))
    Y = np.vstack((Y_act, Y_no_act))
    
    X, Y = eliminar_nan_y_filas_vacias(X, Y)
    
    
    del X_act, X_no_act, Y_act, Y_no_act
    
    return shuffle(X, Y)