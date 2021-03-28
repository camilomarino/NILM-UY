import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta


def visualizar(datos:dict, x:np, y:np, elec:str):
    '''
    Muestra dado una muestra de X e y, se busca en datos el tramo de dataframe
    correspondiente a esa muestra y se grafica mediante matplotlib

    Parameters
    ----------
    datos : dict
        Diccionario de datos.
    x : np
        Muesta de x(fila de la matriz de datos).
    y : np
        Muesta de y(fila de la matriz de datos).
    elec : str
        Nombre del electrodomestico.

    Returns
    -------
    None.
    
    Ejemplo
    -------
        uk.visualizar(datos_ini, x_train[tipo_red][elec][i], 
                      y_train[tipo_red][elec][i], elec)

    '''
    if y.shape[0] == 3:
        tipo_red = 'rectangulos'
    else:
        tipo_red = 'autoencoder'
        
    num_casa = x[-3] #casa del electrodomestico
    # lo siguiente va en formato fecha
    inicio = pd.Timestamp(x[-2], tz='Europe/London') #inicio de la activacion/ no activacion
    fin = pd.Timestamp(x[-1], tz='Europe/London') #fin de la activacion/ no activacion
    plt.figure()
    # Grafico serie agregada y desagregada
    plt.plot(datos[num_casa]['aggregate'][inicio-timedelta(minutes=30):fin+timedelta(minutes=30)])
    plt.plot(datos[num_casa][elec][inicio-timedelta(minutes=30):fin+timedelta(minutes=30)])

    if tipo_red=='rectangulos':
        d_real = y[2] #pocentaje derecho(o superior inferior) real
        l_real = y[1] #pocentaje izquierdo(o porcentaje inferior) real
        p_real = y[0] #potencia media real (altura del rectangulo)
        # Armo el rectangulo y lo grafico
        largo = fin-inicio
        index_real = pd.date_range(start=inicio, end=fin, freq='6S')
        rectangulo_real = pd.DataFrame(index=index_real, columns=['Potencia'])
        rectangulo_real = rectangulo_real.fillna(0)
        rectangulo_real[inicio+l_real*largo:inicio+d_real*largo]=p_real 
        plt.plot(rectangulo_real, alpha=0.5)
        if d_real==0 and l_real==0 and p_real==0:
            plt.title(f'No Activacion Rectangulos {elec}')
        else:
            plt.title(f'Activacion Rectangulos {elec}')
        plt.axvline(inicio, ls='-.',c='black', alpha=0.7)
        plt.axvline(fin, ls='-.',c='black', alpha=0.7)
        plt.ylim(-100,3300)
        plt.legend(['Agregada  '+str(np.int(np.sum(datos[num_casa]['aggregate'][inicio+l_real*largo:inicio+d_real*largo].values))), 
                    'Desagregada  '+str(np.int(np.sum(datos[num_casa][elec][inicio+l_real*largo:inicio+d_real*largo].values))),
                    'Rectangulo Real  '+str(np.int(np.sum(rectangulo_real[inicio+l_real*largo:inicio+d_real*largo].values))), 
                    'Inicio ventana de la red',
                    'Fin ventana de la red'])#'Rectangulo Predict'])
    
    elif tipo_red=='autoencoder':
        # Grafico el target del autoencoder
        index_real = pd.date_range(start=inicio, end=fin, freq='6S')[:-1] # vector de timestamps para el eje X del target
        plt.plot(index_real, y, ls='-.', c='black')
        
        plt.axvline(inicio, ls='-.',c='black', alpha=0.7)
        plt.axvline(fin, ls='-.',c='black', alpha=0.7)
        
        if np.sum(y) == 0:
            plt.title(f'No Activacion Autoencoder {elec}')
        else:
            plt.title(f'Activacion Autoencoder {elec}')
    
        plt.legend(['Agregada', 
                    'Desagregada',
                    'Target autoencoder',
                    'Inicio ventana de la red',
                    'Fin ventana de la red'])
        
        