import os

def crear_estructura_carpetas_pesos(ruta_base:str = ""):
    '''      
    Parameters
    ----------
    ruta_base : str, optional
        Ruta sobre la cual se creara el arbol de carpetas. The default is "".

    Returns
    -------
    None.

    '''
    for data in ['data', 'data_hf']:
         for tipo_red in ['autoencoder', 'rectangulos']:
             for elec in ['kettle', 'fridge', 'washing', 'microwave', 'dish']:
                 ruta_data = os.path.join(ruta_base, data)
                 ruta_red = os.path.join(ruta_data, tipo_red)
                 ruta_elec = os.path.join(ruta_red, elec)
             
                 try: os.makedirs(ruta_data)
                 except: pass
                 try: os.makedirs(ruta_red)
                 except: pass
                 try: os.makedirs(ruta_elec)
                 except: pass