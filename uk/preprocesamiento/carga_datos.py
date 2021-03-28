import pandas as pd
import os
from datetime import timedelta
from tqdm import tqdm
import copy

def carga_datos_desde_hdf(ruta:str, verbose=False) -> dict:
    ''' Dada la ruta del archivo hdf de UK-Dale retorna un 
    diccionario de diccionarios de series de pandas. 
    Las series de pandas tienen indice la fecha en UTF-8
    
    Parametros
    ----------
        ruta : Ruta del archivo ukdale.h5 descaragdo de: https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated
        
    Retorno
    -------
        dict : 
            Retorna un diccionario de diccionarios de series de pandas. 
            El primer indice representa el numero de casa, pudiendo ser:
                1
                2
                3
                4
                5
            mientras que el segundo indice representa el electrodomestico de
            interes o serie agregada, pudiendo ser:
                aggregate
                kettle
                fridge
                microwave
                dish
                washing
            
            
                
    Ejemplo
    -------
        datos_uk = uk.carga_datos_from_hdf('ruta')
        datos_casa1_kettle = datos_uk[1]['kettle'] #Accedo al dataframe de
                                                    la serie de la heladera
                                                    de la casa 1
    '''
    if not isinstance(ruta, str):
        raise TypeError('El parametro debe ser un string de la ruta del archivo')
    if not os.path.isfile(ruta):
        raise FileNotFoundError('Archivo no encontrado')
    
    if verbose: print('Comienza carga de datos')
    h5 = pd.HDFStore(ruta)
    
    
    aggregate_1 = h5['/building1/elec/meter1']
    kettle_1 = h5['/building1/elec/meter10']
    fridge_1 = h5['/building1/elec/meter12']
    washing_1 = h5['/building1/elec/meter5']
    microwave_1 = h5['/building1/elec/meter13']
    dish_1 = h5['/building1/elec/meter6']
    casa_1 = {"aggregate": aggregate_1, "kettle":kettle_1, "fridge": fridge_1, 
              "washing": washing_1, "microwave": microwave_1, "dish":dish_1, 
              "num_casa": 1}
    del aggregate_1
    del kettle_1
    del fridge_1
    del washing_1
    del microwave_1
    del dish_1
    
    aggregate_2 = h5['/building2/elec/meter1']
    kettle_2 = h5['/building2/elec/meter8']
    fridge_2 = h5['/building2/elec/meter14']
    washing_2 = h5['/building2/elec/meter12'] 
    microwave_2 = h5['/building2/elec/meter15']
    dish_2 = h5['/building2/elec/meter13']    
    casa_2 = {"aggregate": aggregate_2, "kettle":kettle_2, 
              "fridge": fridge_2, "washing": washing_2,
             "microwave": microwave_2, "dish":dish_2, "num_casa": 2}
    del aggregate_2
    #del mains_2
    del kettle_2
    del fridge_2
    del microwave_2
    del dish_2
    
    aggregate_3 = h5['/building3/elec/meter1']
    kettle_3 = h5['/building3/elec/meter2']
    #fridge_3 = h5['/building3/elec/meter12'] No hay heladera etiquetada en esta casa
    #washing_3 = h5['/building3/elec/meter5'] No hay lavarropa en esta casa
    #microwave_3 = h5['/building3/elec/meter13'] No hay en esta casa
    #dish_3 = h5['/building3/elec/meter6'] No hay en esta casa
    casa_3 = {"aggregate": aggregate_3, "kettle":kettle_3, "num_casa": 3}#, "fridge": fridge_3, "washing": washing_3, "microwave": microwave_3, "dish":dish_3}
    del aggregate_3
    del kettle_3
    
    
    aggregate_4 = h5['/building4/elec/meter1']
    kettle_4 = h5['/building4/elec/meter3']  #Kettle-Radio
    fridge_4 = h5['/building4/elec/meter5']  #Freezer
    #washing_4 = h5['/building4/elec/meter5']  No hay
    #microwave_4 = h5['/building4/elec/meter13']  No hay
    #dish_4 = h5['/building4/elec/meter6']  No hay
    casa_4 = {"aggregate": aggregate_4, "num_casa": 4, "kettle":kettle_4, "fridge": fridge_4}# "washing": washing_4, "microwave": microwave_4, "dish":dish_4}
    del aggregate_4
    del kettle_4
    del fridge_4
    
    aggregate_5 = h5['/building5/elec/meter1']
    kettle_5 = h5['/building5/elec/meter18'] 
    fridge_5 = h5['/building5/elec/meter19'] 
    washing_5 = h5['/building5/elec/meter24'] #washer dryer
    microwave_5 = h5['/building5/elec/meter23'] 
    dish_5 = h5['/building5/elec/meter22']
    #mains_5 = load_mains("C:/Users/Camilo/Documents/mains5.dat")
    casa_5 = {"aggregate": aggregate_5, "washing": washing_5, 
              "num_casa": 5, "kettle": kettle_5, "fridge": fridge_5, 
              "microwave": microwave_5, "dish": dish_5}
    
    del aggregate_5
    del kettle_5
    del fridge_5
    del washing_5
    del microwave_5 
    del dish_5
    
    
    datos = {1: casa_1, 2: casa_2, 3: casa_3, 4: casa_4, 5: casa_5}
    if verbose: print('Fin carga de datos')
    if verbose: print('-'*40)
    return datos


def reindex(dataframe, index_0, index_n, freq="6S", fill_value=0, limit=30):
    '''
    Dado un dataframe e indices de inicio y fin, reindexa la serie rellenando
    datos faltantes con 0
    '''
    idx = pd.date_range(index_0, index_n, freq=freq)
    dataframe_ = dataframe.reindex(
        idx, method="pad", fill_value=fill_value, limit=limit
    )
    return dataframe_


def separacion_datos(datos:dict, dias_final:int=14, min_pad:int=180, 
                     columna_referencia:str="aggregate", verbose:bool=False):
    ''' Dado el diccionario de datos y los parametros, se divide el 
    diccionario en 2, uno para las ultimas X dias de datos y otro para el
    resto.
    Notar que por defecto se toman las ultimas 2 semanas en base a la serie
    agregada. Esta nota es debido a que todos los electrodomesticos no 
    necesariamente tienen el mismo rangoEjemplo
    -------
        ruta = '/home/camilo/base_de_datos_nilm/ukdale.h5'
        datos = uk.carga_datos_desde_hdf(ruta, verbose=True)
        datos_ini, datos_fin = uk.separacion_datos(datos, dias_final=14, verbose=True)
         de fechas.
    Nota: se extiende con 0's la serie durante 3 horas, para mitigar efectos
          de borde
    Note: Se necesitan al menos 10/12 Gb de ram para este proceso
    
    Parametros
    ----------
        datos : diccionario de dataframes
        dias_final : cuantos dias se tomaran desde el final para separar los
                     datos
        min_pad : 
        columna_referencia : nombre de electrodomestico o serie agregada
                            que se usara como base del reindexado
            
    Retorno
    -------
        datos_ini : diccionario de las primeras semanas
        datos_fin : dicccionario de los ultimos dias_final
        
    Ejemplo
    -------
        ruta = '/home/camilo/base_de_datos_nilm/ukdale.h5'
        datos = uk.carga_datos_desde_hdf(ruta, verbose=True)
        datos_ini, datos_fin = uk.separacion_datos(datos, dias_final=14, verbose=True)
        
    '''
    if verbose: print("Comienza separacion de datos")
    casa_ini = copy.deepcopy(datos)
    casa_fin = copy.deepcopy(datos)
    
    for num_casa in [1,2,3,4,5]:
        casa_ini[num_casa]['num_casa'] = num_casa
        casa_fin[num_casa]['num_casa'] = num_casa
        
        
        inicio_agre = datos[num_casa][columna_referencia].index[0]
        fin_agre = datos[num_casa][columna_referencia].index[-1]
    
        elecs = list(datos[num_casa].keys())
        elecs.remove("num_casa")
        for elec in elecs:
            casa_ini[num_casa][elec] = reindex(
                datos[num_casa][elec], inicio_agre, fin_agre - timedelta(days=dias_final))
            
            casa_ini[num_casa][elec] = reindex(
                datos[num_casa][elec],
                inicio_agre - timedelta(minutes=min_pad),
                fin_agre - timedelta(days=dias_final) + timedelta(minutes=min_pad),)
    
            casa_fin[num_casa][elec] = reindex(
                datos[num_casa][elec], fin_agre - timedelta(days=dias_final), fin_agre)
            
            casa_fin[num_casa][elec] = reindex(
                datos[num_casa][elec],
                fin_agre - timedelta(days=dias_final) - timedelta(minutes=min_pad),
                fin_agre + timedelta(minutes=min_pad),)
        
    if verbose: print('Fin separacion de datos')
    if verbose: print('-'*40)    
    return casa_ini, casa_fin



