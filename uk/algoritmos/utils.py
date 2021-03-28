import numpy as np

def normalize_X(X, std):
    """
    Se dividie entre la media de cada secuencia y se divide por una desviacion
    standar del conjunto de train
    """
    return (X - np.expand_dims(np.mean(X, axis=1), axis=2)) / std


def normalize_Y(Y, potencia_maxima_train, red="rectangulos"):
    """
    Se divide por la potencia maxima de train
    """
    if red == "rectangulos":
        Y = Y.copy()
        Y[:, 0] = Y[:, 0] / potencia_maxima_train
        return Y

    elif red == "autoencoder":
        return Y / potencia_maxima_train

    else:
        raise ValueError('Red debe ser autoencoder o rectangulos')


def unnormalize_Y(Y, potencia_maxima_train, red):
    """
    Se multiplica por la potencia maxima de train
    """
    if red == "rectangulos":
        Y = Y.copy()
        Y[:, 0] = Y[:, 0] * potencia_maxima_train
        return Y

    elif red == "autoencoder":
        return Y * potencia_maxima_train

    else:
        raise ValueError('Red debe ser autoencoder o rectangulos')
        
        
def salida_rectangulos_to_serie_numpy(x:np.ndarray, size: int) -> np.ndarray:
    '''
    Trnsforma un vector de tamaño 3, que indican potencia media del rectangulo
    en todo el size, porcentaje de inicio y fin.

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION.

    Returns
    -------
    Un numpy array que  es un rectangulo.

    '''
    out = np.zeros(size)
    out[int(max(0,round(x[1]*size))):int(min(round(x[2]*size), size))] = x[0]
    return out