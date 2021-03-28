import numpy as np
from datetime import timedelta
from sklearn.utils import shuffle
from .crea_X_Y_utils import activation_series_for_chunk_h5, eliminar_nan_y_filas_vacias

#%%
def crear_vectores_sinteticos(datos:dict, 
                              electrodomestico_base:str,
                              electrodemsticos_distractores:list,
                              numeros_casas:list,
                              parametros:dict, 
                              largo_de_ventana:int,
                              tipo_red:str,
                              cant_a_calcular:int = 100,
                              prob_distractores:float = 0.3, 
                              seed:bool = None,
                              verbose:bool = False):
    '''
    Notas
    -----
    Nota1:
        Se utilizan las mismas casas para el electrodomestico base que para los
        electrodomesticos distractores. 
        Para cada muestra a calcular(fila de x)
        que sea una activacion, se sorte con prob_distractores para cada otra lsita
        de electrodomesticos si debe aparecer o no una activacion de este 
        electrodomestico. Si debe aparecer, se sortea entre todos las activaciones
        de ese electrodomestico, cual de ellas sera colocada. Por ultimo se sortea
        una posicion para esta activacion distractora. 
        Notar que son calculadas todas las acticaciones de todas las casas, por lo 
        cual si se usa UK-Dale puro, se generara una base desbalanceada hacia la 
        casa 1.
    
    Nota2:
        Se implementa este codigo para adaptar el desarrollado  en el proyecto,
        el cual es bastante desprolijo. Se busca dar una interfaz entendible a
        la funcion crear_datos_sinteticos.
    
    
    Parameters
    ----------
    datos : dict
        Datos obtenidos mediante la separacion de datos
    electrodomestico_base : str
        Nombre de electrodomestico base sobre el que se calculara los x e y.
    electrodemsticos_distractores : list
        Lista de electrodomesticos que se usaran como distractores.
    numeros_casas : list
        Casas a utilizar para generar los datos.
    parametros : dict
        Parametros de todos los electrodomesticos que se usaran.
    largo_de_ventana : int
        Largo de la ventana.
    cant_a_calcular : TYPE, optional
        Numero de vectores a calcular(alto de x). Se calcula misma cantidad de
        activaciones que no activaciones
    prob_distractores : TYPE, optional
        Probabilidad de incluir de que el vector tenga cada uno de los demas
        electrodomesticos distractores. The default is 0.3.
    tipo_red : str, optional
        Tipo de red sobre la que se calcula los datos sinteticos.
        The default is 'rectangulos'.
    seed : bool, optional
        Semilla. The default is None.
    verbose : TYPE, optional
        Modo verboso. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.
        
    Ejemplo
    -------
        x_r, y_r = uk.crear_vectores_sinteticos(datos = casas_ini, 
                                   electrodomestico_base = 'kettle', 
                                   electrodemsticos_distractores = ['fridge', 
                                                                    'dish', 
                                                                    'microwave','
                                                                    washing'], 
                                   numeros_casas = [1,2,3,4,5], 
                                   parametros = parametros,
                                   largo_de_ventana = win_len['kettle'],
                                   tipo_red = 'rectangulos',
                                   cant_a_calcular = 1000,
                                   prob_distractores = 0.5, 
                                   seed = 5,
                                  )

    '''
    if not seed is None:np.random.seed(seed)
    activaciones_base = []
    activaciones_distractoras = [ [], [], [], [] ]
    
    for num_casa in numeros_casas:
        activaciones_base +=(
                activation_series_for_chunk_h5(datos[num_casa][electrodomestico_base],
                                               **(parametros[electrodomestico_base]))
                            )
        
        for i, elec in enumerate(electrodemsticos_distractores):
            if elec in datos[num_casa].keys():
                activaciones_distractoras[i] += (
                        activation_series_for_chunk_h5(datos[num_casa][elec],
                                                       **(parametros[elec]))
                                                    )
                                                    
    #import ipdb; ipdb.set_trace()
    x, y_rect, y_auto= crear_datos_sinteticos(activaciones_base,
                               activaciones_distractoras,
                               timedelta(minutes=largo_de_ventana),
                               activaciones_a_calcular = cant_a_calcular//2,
                               no_activaciones_a_calcular = cant_a_calcular//2,
                               p = prob_distractores,
                               seed = seed,
                               )
    if tipo_red== 'rectangulos':
        x, y_rect = eliminar_nan_y_filas_vacias(x, y_rect)
        return shuffle(x, y_rect)
    elif tipo_red=='autoencoder':
        x, y_auto = eliminar_nan_y_filas_vacias(x, y_auto)
        return shuffle(x, y_auto)
    
def crear_datos_sinteticos(activaciones_base:list, 
                           activaciones_distractoras:list,
                           windows_len,
                           activaciones_a_calcular=10,
                           no_activaciones_a_calcular=10,
                           cantidad_muestras=500,
                           p=0.3,
                           seed = None):
    '''Input:
        serie_agregada: Dataframe de la serie agregada
        activaciones_base: Lista de listas. Activaciones del electrodomestico del 
                        que se quieren generar datos  sinteticos
        
        activaciones_i(i dinstinto de 0): Lista de listas. Activaciones de los electrodemsticos 
                        que no son del electrodomestico de interes
        
        
                       Nota:  La lista mas interna contendra 
                              dataframes de activaciones
        windows_len: tamano de la ventana en  formato tiempo
        
        
        Output:
            X: numpy array de la serie agregada (los metadatos son -1)
            y: numpy array, son las etiquetas de la red de rectangulos
        
        '''
        
    if not seed is None:np.random.seed(seed)
    size_windows_len = int(windows_len.total_seconds()/6)
    X = np.zeros((activaciones_a_calcular+no_activaciones_a_calcular, size_windows_len))
    X_metadatos = np.ones((activaciones_a_calcular+no_activaciones_a_calcular, 3))*(-1) #Son negativos indicando 
                                            # que no hay por ser datos sinteticos
    Y_rectangulos = np.zeros((activaciones_a_calcular+no_activaciones_a_calcular, 3))
    
    Y_autoencoder = np.zeros((activaciones_a_calcular+no_activaciones_a_calcular, 
                              size_windows_len))
    
    cantidad_activaciones_base = len(activaciones_base)        
    
    
    #for i in range(cantidad_muestras):
    for i in range(activaciones_a_calcular):    
        largo_verifica = True
        while largo_verifica==True:
            numero_de_activacion = np.random.randint(0,cantidad_activaciones_base)
            size_activacion = len(activaciones_base[numero_de_activacion])
            
            # Ahora se debe calcular en que lugar de la ventana se colocara la 
            # activacion
            if size_activacion <= size_windows_len:
                diferencia_size = (size_windows_len - 
                                   size_activacion)
                largo_verifica = False
        
        corrimiento = np.random.randint(0, diferencia_size+1)
        
        
        X[i][corrimiento : corrimiento+size_activacion] += \
                                activaciones_base[numero_de_activacion].to_numpy()[:,0]
        
        porcentaje_inicial = corrimiento/size_windows_len
        porcentaje_final = (corrimiento+size_activacion)/size_windows_len
        potencia = np.mean(activaciones_base[numero_de_activacion].to_numpy()[1:-1,0] )
        Y_rectangulos[i] = np.array([potencia, porcentaje_inicial, porcentaje_final])
        
        Y_autoencoder[i] = X[i]
        
        for activaciones in activaciones_distractoras:
            cantidad_activaciones = len(activaciones)
            if cantidad_activaciones==0:
                break
            if np.random.binomial(1,p)==True: 
                    
                numero_de_activacion = np.random.randint(0,cantidad_activaciones)
                size_activacion = len(activaciones[numero_de_activacion])
                corrimiento = np.random.randint(-size_activacion , size_activacion)
                
                activacion = activaciones[numero_de_activacion].to_numpy()[:,0]
                activacion_pad = np.pad(activacion, (size_activacion+size_windows_len, size_activacion+size_windows_len))
                activacion = activacion_pad[(size_activacion+size_windows_len+
                                             corrimiento):(size_activacion+size_windows_len+
                                                            corrimiento+
                                                            size_windows_len)]
                
                X[i] += activacion
        
        
        
    for i in range(activaciones_a_calcular, activaciones_a_calcular+no_activaciones_a_calcular):
        for activaciones in activaciones_distractoras:
            cantidad_activaciones = len(activaciones) 
            if cantidad_activaciones==0:
                break   
            if np.random.binomial(1,p)==True: 
                
                numero_de_activacion = np.random.randint(0,cantidad_activaciones)
                size_activacion = len(activaciones[numero_de_activacion])
                corrimiento = np.random.randint(-size_activacion , size_activacion)
                
                activacion = activaciones[numero_de_activacion].to_numpy()[:,0]
                activacion_pad = np.pad(activacion, (size_activacion+size_windows_len, size_activacion+size_windows_len))
                activacion = activacion_pad[(size_activacion+size_windows_len+
                                             corrimiento):(size_activacion+size_windows_len+
                                                            corrimiento+
                                                            size_windows_len)]
                X[i] += activacion
        
        
    X = np.hstack((X, X_metadatos))   


    return X, Y_rectangulos, Y_autoencoder


