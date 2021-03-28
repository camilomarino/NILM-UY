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

# %%
import uk
# %matplotlib
import matplotlib.pyplot as plt

# %%
#cargo unos pickles levemente procesados de antes, los reproeso
a = uk.cargar('/home/camilo/repos_fing/base_de_datos_nilm/pablo1_dict.pkl')
b = uk.cargar('/home/camilo/repos_fing/base_de_datos_nilm/pablo2_dict.pkl')
c = uk.cargar('/home/camilo/repos_fing/base_de_datos_nilm/cardal_dict.pkl')

# %%
import numpy as np
def cut_serie(a):
    '''
    elimino los posibles nan al prinicipio y fin
    '''
    for electrodomestico in a[6]: 
        if electrodomestico=='num_casa':
            continue
        inicio, fin = a[6]['power'].index[0], a[6]['power'].index[-1]

        i = 1
        while np.isnan(a[6]['power'][inicio]):
            inicio = a[6]['power'].index[i]
            i += 1
        
        i = -2
        while np.isnan(a[6]['power'][fin]):
            fin = a[6]['power'].index[i]
            i -= 1
            
        a[6][electrodomestico] = a[6][electrodomestico][inicio:fin]
        

def cut_first(a, x):
    '''
    elimina los primeros x muestras de todos los electrodomesticos
    '''
    for electrodomestico in a[6]: 
        if electrodomestico=='num_casa':
            continue
        a[6][electrodomestico] = a[6][electrodomestico][x:] 

def view_serie(a):
    '''
    muestra la serie con plot
    '''
    plt.figure()
    for electrodomestico in a[6].keys():
        if electrodomestico=='form factor' or electrodomestico=='phase' or electrodomestico=='num_casa':
            continue
        plt.plot(a[6][electrodomestico], label=electrodomestico)
    plt.legend()


# %%
cut_serie(a); cut_serie(b); cut_serie(c);   

# %%
cut_first(a, 5800)

# %%
view_serie(a)

# %%
view_serie(b)

# %%
view_serie(c)

# %%
# Lo llevo a una estructura de como la de uk-dale
datos_uru = {6: a[6],
             7: b[6],
             8: c[6]}


# %%
def rename_power2aggregate(datos):
    '''
    cambia el nombre de la key de 'power' a 'aggregate' para compatibilidad con lo de kelly
    '''
    for key in datos:
        datos[key]['aggregate'] = datos[key]['power']
        del datos[key]['power']
rename_power2aggregate(datos_uru)

# %%
datos_uru

# %%
uk.guardar(datos_uru, 'datos_uruguay.pickle')
