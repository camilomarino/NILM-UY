import pickle
import os
import numpy as np

def guardar(obj:object, name:str):
    '''
    Parameters
    ----------
    obj : TYPE
        Objeto a guardar.
    name : str
        NOmbre del archivo a ser guardado.

    Ejemplo
    -------
        uk.guardar(datos_ini, 'datos_ini.pickle')
        uk.guardar(datos_fin, 'datos_fin.pickle')

    '''
    
    with open(name, "wb") as f:
        pickle.dump(obj, f, protocol=3)


def cargar(name:str):
    '''
    
    Parameters
    ----------
    name : str
        Nombre del objeto a cargar.

    Returns
    -------
    TYPE
        Objeto cargado.
        
    Ejemplo
    -------
        datos_ini = uk.cargar('datos_ini.pickle')
        datos_fin = uk.cargar('datos_fin.pickle')

    '''
    with open(name, "rb") as f:
        return pickle.load(f)
    
    
def crear_estructura_carpetas(ruta_base:str = ""):
    '''
    Se crea la estructura completa de carpetas a utilizar de la forma:
        ruta_base/
        ├── data
        │   ├── autoencoder
        │   │   ├── dish
        │   │   │   ├── test_no_visto
        │   │   │   ├── test_visto
        │   │   │   ├── train
        │   │   │   └── validacion
        │   │   ├── fridge
        │   │   │   ├── test_no_visto
        │   │   │   ├── test_visto
        │   │   │   ├── train
        │   │   │   └── validacion
        │   │   ├── kettle
        │   │   │   ├── test_no_visto
        │   │   │   ├── test_visto
        │   │   │   ├── train
        │   │   │   └── validacion
        │   │   ├── microwave
        │   │   │   ├── test_no_visto
        │   │   │   ├── test_visto
        │   │   │   ├── train
        │   │   │   └── validacion
        │   │   └── washing
        │   │       ├── test_no_visto
        │   │       ├── test_visto
        │   │       ├── train
        │   │       └── validacion
        │   └── rectangulos
        │       ├── dish
        │       │   ├── test_no_visto
        │       │   ├── test_visto
        │       │   ├── train
        │       │   └── validacion
        │       ├── fridge
        │       │   ├── test_no_visto
        │       │   ├── test_visto
        │       │   ├── train
        │       │   └── validacion
        │       ├── kettle
        │       │   ├── test_no_visto
        │       │   ├── test_visto
        │       │   ├── train
        │       │   └── validacion
        │       ├── microwave
        │       │   ├── test_no_visto
        │       │   ├── test_visto
        │       │   ├── train
        │       │   └── validacion
        │       └── washing
        │           ├── test_no_visto
        │           ├── test_visto
        │           ├── train
        │           └── validacion
        └── data_hf
            ├── autoencoder
            │   ├── dish
            │   │   ├── test_no_visto
            │   │   ├── test_visto
            │   │   ├── train
            │   │   └── validacion
            │   ├── fridge
            │   │   ├── test_no_visto
            │   │   ├── test_visto
            │   │   ├── train
            │   │   └── validacion
            │   ├── kettle
            │   │   ├── test_no_visto
            │   │   ├── test_visto
            │   │   ├── train
            │   │   └── validacion
            │   ├── microwave
            │   │   ├── test_no_visto
            │   │   ├── test_visto
            │   │   ├── train
            │   │   └── validacion
            │   └── washing
            │       ├── test_no_visto
            │       ├── test_visto
            │       ├── train
            │       └── validacion
            └── rectangulos
                ├── dish
                │   ├── test_no_visto
                │   ├── test_visto
                │   ├── train
                │   └── validacion
                ├── fridge
                │   ├── test_no_visto
                │   ├── test_visto
                │   ├── train
                │   └── validacion
                ├── kettle
                │   ├── test_no_visto
                │   ├── test_visto
                │   ├── train
                │   └── validacion
                ├── microwave
                │   ├── test_no_visto
                │   ├── test_visto
                │   ├── train
                │   └── validacion
                └── washing
                    ├── test_no_visto
                    ├── test_visto
                    ├── train
                    └── validacion

        
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
                 ruta_train = os.path.join(ruta_elec, 'train')
                 ruta_validacion = os.path.join(ruta_elec, 'validacion')
                 ruta_test_no_visto = os.path.join(ruta_elec, 'test_no_visto')
                 ruta_test_visto = os.path.join(ruta_elec, 'test_visto')
                 ruta_sintetico = os.path.join(ruta_elec, 'sintetico')
             
                 try: os.makedirs(ruta_data)
                 except: pass
                 try: os.makedirs(ruta_red)
                 except: pass
                 try: os.makedirs(ruta_train)
                 except: pass
                 try: os.makedirs(ruta_elec)
                 except: pass
                 try: os.makedirs(ruta_validacion)
                 except: pass
                 try: os.makedirs(ruta_test_no_visto)
                 except: pass
                 try: os.makedirs(ruta_test_visto)
                 except: pass
                 try: os.makedirs(ruta_sintetico)
                 except: pass
             
def guardar_X_y(x:dict, y:dict, tipo_data, conjunto, ruta_base=""):
    if (conjunto!='test_no_visto' and conjunto!='test_visto' and 
        conjunto!='train' and conjunto!='validacion' and
        conjunto!='sintetico'):
        raise ValueError("El valor de conjunto no es valido")
    if tipo_data!='data' and tipo_data!='data_hf':
        raise ValueError("El valor de tipo_data no es valido")
        
    
    for tipo_red in x.keys():
        for elec in x[tipo_red].keys():
            ruta = os.path.join(ruta_base, tipo_data, tipo_red, elec, conjunto)
            np.save(os.path.join(ruta, 'x.npy'), x[tipo_red][elec])
            np.save(os.path.join(ruta, 'y.npy'), y[tipo_red][elec])
                 
def cargar_X_y(tipo_data, conjunto, ruta_base=""):
    if (conjunto!='test_no_visto' and conjunto!='test_visto' and 
        conjunto!='train' and conjunto!='validacion' and
        conjunto!='sintetico'):
        raise ValueError("El valor de conjunto no es valido")
    if tipo_data!='data' and tipo_data!='data_hf':
        raise ValueError("El valor de tipo_data no es valido")
            
    x = {'autoencoder' : {'kettle' : None,
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
    y = {'autoencoder' : {'kettle' : None,
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
    
    for tipo_red in ['autoencoder', 'rectangulos']:
        for elec in ['kettle', 'fridge', 'washing', 'microwave', 'dish']:
            ruta = os.path.join(ruta_base, tipo_data, tipo_red, elec, conjunto)
            try:
                x[tipo_red][elec] = np.load(os.path.join(ruta, 'x.npy'))
                y[tipo_red][elec] = np.load(os.path.join(ruta, 'y.npy'))
            except ValueError:
                x[tipo_red][elec] = None
                y[tipo_red][elec] = None
    
    return x,y
                 
    