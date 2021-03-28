import numpy as np

def cantidad_por_casa(x:np, y:np):
    '''
    Hace print de las cantidades de una pareja x y
    Parameters
    ----------
    x : TYPE
        vector numpy de X.
    y : TYPE
        vector numpy de y.

    Returns
    -------
    None.

    '''
    cant_por_casa = {}
    for num_casa in np.unique(x[:,-3]):
        cant_por_casa[num_casa] = np.sum(x[:, -3]==num_casa)
        print(f"casa: {int(num_casa)}\tcantidad: {cant_por_casa[num_casa]}")
        print("-"*50)
        y_casa = y[(x[:, -3]==num_casa), :]
        # import ipdb; ipdb.set_trace()
        num_activaciones = np.sum(np.sum(y_casa, axis=1)==0)
        print(f"\t\t\tactivaciones: \t\t{num_activaciones}")
        print(f"\t\t\tno activaciones: \t{len(y_casa)-num_activaciones}\n\n\n")

