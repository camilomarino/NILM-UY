import numpy as np
from datetime import timedelta
import pandas as pd

# Activaciones para datos con indice timestamp
def activation_series_for_chunk_h5(
    chunk, min_off_duration=0, min_on_duration=0, border=0, on_power_threshold=5
):
    """Returns runs of an appliance.
    Most appliances spend a lot of their time off.  This function finds
    periods when the appliance is on.
    Parameters
    ----------
    chunk : pd.Series
    min_off_duration : int
        If min_off_duration > 0 then ignore 'off' periods less than
        min_off_duration seconds of sub-threshold power consumption
        (e.g. a washing machine might draw no power for a short
        period while the clothes soak.)  Defaults to 0.
    min_on_duration : int
        Any activation lasting less seconds than min_on_duration will be
        ignored.  Defaults to 0.
    border : int
        Number of rows to include before and after the detected activation
    on_power_threshold : int or float
        Defaults to self.on_power_threshold()
    Returns
    -------
    list of pd.Series.  Each series contains one activation.
    """
    when_on = chunk >= on_power_threshold

    # Find state changes
    state_changes = when_on.astype(np.float).diff()
    del when_on
    switch_on_events = np.where(state_changes == 1)[0]
    switch_off_events = np.where(state_changes == -1)[0]
    del state_changes

    if len(switch_on_events) == 0 or len(switch_off_events) == 0:
        return []

    # Make sure events align
    if switch_off_events[0] < switch_on_events[0]:
        switch_off_events = switch_off_events[1:]
        if len(switch_off_events) == 0:
            return []
    if switch_on_events[-1] > switch_off_events[-1]:
        switch_on_events = switch_on_events[:-1]
        if len(switch_on_events) == 0:
            return []
    assert len(switch_on_events) == len(switch_off_events)

    # Smooth over off-durations less than min_off_duration
    if min_off_duration > 0:
        off_durations = (
            chunk.index[switch_on_events[1:]] - chunk.index[switch_off_events[:-1]]
        )

        off_durations = (
            off_durations
        ).total_seconds()  # timedelta64_to_secs(off_durations)

        above_threshold_off_durations = np.where(off_durations >= min_off_duration)[0]

        # Now remove off_events and on_events
        switch_off_events = switch_off_events[
            np.concatenate(
                [above_threshold_off_durations, [len(switch_off_events) - 1]]
            )
        ]
        switch_on_events = switch_on_events[
            np.concatenate([[0], above_threshold_off_durations + 1])
        ]
    assert len(switch_on_events) == len(switch_off_events)

    activations = []
    for on, off in zip(switch_on_events, switch_off_events):
        duration = (chunk.index[off] - chunk.index[on]).total_seconds()
        if duration < min_on_duration:
            continue
        on -= 1 + border
        if on < 0:
            on = 0
        off += border
        activations.append(chunk.iloc[on:off])

    return activations

def window(win_len, activacion, T=6):
    activacion_inicio = activacion.index[0]  # inicio de la activacion
    activacion_fin = activacion.index[-1]  # fin de la activacion
    activacion_largo = activacion_fin - activacion_inicio  # largo de la activacion

    periodos_activacion = int(activacion_largo.total_seconds() / T)
    periodos_win = int(win_len.total_seconds() / T)

    if periodos_activacion <= periodos_win:
        difrencia_periodos = periodos_win - periodos_activacion

        periodos_corrido = np.random.randint(0, difrencia_periodos + 1)

        inicio = activacion_inicio - periodos_corrido * timedelta(seconds=T)
        fin = inicio + win_len
        porcentaje_inicial = periodos_corrido / (periodos_win - 1)
        porcentaje_final = (periodos_corrido + periodos_activacion) / (periodos_win - 1)
        P = np.mean(activacion[1:-1])

    elif periodos_activacion > periodos_win:
        difrencia_periodos = periodos_activacion - periodos_win

        periodos_corrido = np.random.randint(0, difrencia_periodos + 1)

        inicio = activacion_inicio + periodos_corrido * timedelta(seconds=T)
        fin = inicio + win_len
        porcentaje_inicial = 0
        porcentaje_final = 1
        P = np.mean(activacion[inicio:fin])

    return P, inicio, fin, porcentaje_inicial, porcentaje_final


def window_sin_activacion(win_len, activacion1, activacion2, T=6):
    inicio_activacion_primera = activacion1.index[0]
    inicio_activacion_segunda = activacion2.index[0]
    fin_activacion_primera = activacion1.index[-1]
    fin_activacion_segunda = activacion2.index[-1]

    periodos_win = int(win_len.total_seconds() / T)

    periodo_inicio_primera = int(inicio_activacion_primera.timestamp() / T)
    periodo_inicio_segunda = int(inicio_activacion_segunda.timestamp() / T)
    periodo_fin_primera = int(fin_activacion_primera.timestamp() / T)
    periodo_fin_segunda = int(fin_activacion_segunda.timestamp() / T)

    periodo_inicial = max(
        periodo_fin_primera - periodos_win / 2,
        periodo_inicio_primera + periodos_win / 2,
    )

    periodo_final = min(
        periodo_fin_segunda - periodos_win / 2,
        periodo_inicio_segunda + periodos_win / 2,
    )

    if periodo_inicial <= periodo_final:
        diferencia_periodos = periodo_final - periodo_inicial
        rand_periodo = np.random.randint(0, diferencia_periodos + 1)
        inicio = pd.Timestamp(
            (periodo_inicial + rand_periodo - int(periodos_win / 2)) * T,
            unit="s",
            tz="Europe/London",
        )

        fin = inicio + win_len

    else:
        inicio = timedelta(seconds=0)
        fin = timedelta(seconds=0)

    return inicio, fin



def get_X_y(activaciones, aggregate, win_len, num_casa=0):
    tot_features = int(win_len.total_seconds() / 6)
    x_act = np.empty((len(activaciones), tot_features + 3))
    y_act = np.empty((len(activaciones), 3))

    """
    Por fuerza bruta se hace que se la serie tenga en cuenta sola la primer activacion.
    Se sorttean intervalos de tiempo en el que la actiacion si o si debe ser la primer activacion
    completa.
    """
    for i in range(len(activaciones)):
        # print(i)
        if i != 0:
            ini_act_anterior = activaciones[i - 1].index[0]
        else:
            ini_act_anterior = aggregate.index[0]

        cond = True
        while cond:
            P, inicio, fin, l, r = window(win_len, activaciones[i], T=6)
            if inicio >= ini_act_anterior:
                cond = False

        y_act[i, :] = np.array([P, l, r])
        if len(((aggregate[inicio:fin]).to_numpy())[:, 0]) <= tot_features:
            x_act[i, :-3] = ((aggregate[inicio:fin]).to_numpy())[:, 0]
        else:
            x_act[i, :-3] = ((aggregate[inicio:fin]).to_numpy())[
                0:tot_features, 0
            ]  # elimino casos que pueden quedar de tamano 1 mayor

        x_act[i, -3] = num_casa
        x_act[i, -2] = inicio.value
        x_act[i, -1] = fin.value

    # y_act[:,0] = y_act[:,0]/np.max(y_act[:,0]) #se normaliza la fila de potencias

    # x_act =  x_act - x_act.mean(axis = 1, keepdims=True)

    return x_act, y_act


# Para cada electrodomestico se define una matrix X e Y apropiadas para el autoencoder
# X: cada fila es un cacho de serie que contiene una activacion completa
# Y: cada fila es el cacho de serie desagregada que contiene una activacion completa


def get_X_y_ae(activaciones, aggregate, desagregada, win_len, num_casa=0):
    tot_features = int(win_len.total_seconds() / 6)
    x_act = np.empty((len(activaciones), tot_features + 3))
    y_act = np.zeros((len(activaciones), tot_features))

    for i in range(len(activaciones)):

        """
        Por fuerza bruta se hace que se la serie tenga en cuenta sola la primer activacion.
        Se sorttean intervalos de tiempo en el que la actiacion si o si debe ser la primer activacion
        completa.
        """
        if i != 0:
            ini_act_anterior = activaciones[i - 1].index[0]
        else:
            ini_act_anterior = aggregate.index[0]

        cond = True
        while cond:
            P, inicio, fin, l, r = window(win_len, activaciones[i], T=6)
            if inicio >= ini_act_anterior:
                cond = False

        primera_activacion = desagregada[inicio:fin] * 0
        primera_activacion[
            activaciones[i].index[0] : activaciones[i].index[-1]
        ] = activaciones[i]

        if len(((desagregada[inicio:fin]).to_numpy())[:, 0]) <= tot_features:
            y_act[i, :] = ((primera_activacion).to_numpy())[:, 0]
        else:
            y_act[i, :] = ((primera_activacion).to_numpy())[
                0:tot_features, 0
            ]  # elimino casos que pueden quedar de tamano 1 mayor

        if len(((aggregate[inicio:fin]).to_numpy())[:, 0]) <= tot_features:
            x_act[i, :-3] = ((aggregate[inicio:fin]).to_numpy())[:, 0]
        else:
            x_act[i, :-3] = ((aggregate[inicio:fin]).to_numpy())[
                0:tot_features, 0
            ]  # elimino casos que pueden quedar de tamano 1 mayor

        x_act[i, -3] = num_casa
        x_act[i, -2] = inicio.value
        x_act[i, -1] = fin.value

    # y_act[:,0] = y_act[:,0]/np.max(y_act[:,0]) #se normaliza la fila de potencias

    # x_act =  x_act - x_act.mean(axis = 1, keepdims=True)

    return x_act, y_act

# Para cada electrodomestico se define una matrix X e Y
# X: cada fila es un cacho de serie que contiene una activacion completa
# Y: cada fila es una terna de P, l, r
def get_X_y_no_act(activaciones, aggregate, win_len, num_casa=0):
    window_no_activaciones = []
    for i in range(len(activaciones) - 1):
        inicio, fin = window_sin_activacion(
            win_len, activaciones[i], activaciones[i + 1], T=6
        )
        if not (inicio == timedelta(seconds=0) and fin == timedelta(seconds=0)):
            window_no_activaciones.append([inicio, fin])

        else:
            cond = True
            iteraciones = 0
            while cond:
                j = np.random.randint(0, len(activaciones) - 1)
                inicio, fin = window_sin_activacion(
                    win_len, activaciones[j], activaciones[j + 1], T=6
                )
                iteraciones += 1
                if iteraciones >= 5:
                    cond = False
                    # print('loop')
                if not (inicio == timedelta(seconds=0) and fin == timedelta(seconds=0)):
                    window_no_activaciones.append([inicio, fin])
                    cond = False

    tot_features = int(win_len.total_seconds() / 6)
    x_no_act = np.empty((len(window_no_activaciones), tot_features + 3))
    y_no_act = np.zeros((len(window_no_activaciones), 3))

    for i in range(len(window_no_activaciones)):
        inicio = window_no_activaciones[i][0]
        fin = window_no_activaciones[i][1]

        x_no_act[i, -3] = num_casa
        x_no_act[i, -2] = inicio.value
        x_no_act[i, -1] = fin.value
        if len(((aggregate[inicio:fin]).to_numpy())[:, 0]) <= tot_features:
            x_no_act[i, :-3] = ((aggregate[inicio:fin]).to_numpy())[:, 0]
        else:
            x_no_act[i, :-3] = ((aggregate[inicio:fin]).to_numpy())[
                0:tot_features, 0
            ]  # elimino casos que pueden quedar de tamano 1 mayor

    # x_no_act =  x_no_act - x_no_act.mean(axis = 1, keepdims=True)

    return x_no_act, y_no_act


# Para cada electrodomestico se define una matrix X e Y
# X: cada fila es un cacho de serie que contiene una activacion completa
# Y: cada fila es una terna de P, l, r
def get_X_y_no_act_ae(activaciones, aggregate, desaggregate, win_len, num_casa=0):

    window_no_activaciones = []
    for i in range(len(activaciones) - 1):
        inicio, fin = window_sin_activacion(
            win_len, activaciones[i], activaciones[i + 1], T=6
        )
        if not (inicio == timedelta(seconds=0) and fin == timedelta(seconds=0)):
            window_no_activaciones.append([inicio, fin])
        else:
            cond = True
            while cond:
                j = np.random.randint(1, len(activaciones) - 2)
                inicio, fin = window_sin_activacion(
                    win_len, activaciones[j], activaciones[j + 1], T=6
                )
                if not (inicio == timedelta(seconds=0) and fin == timedelta(seconds=0)):
                    window_no_activaciones.append([inicio, fin])
                    cond = False

    tot_features = int(win_len.total_seconds() / 6)
    x_no_act = np.empty((len(window_no_activaciones), tot_features + 3))
    y_no_act = np.zeros((len(window_no_activaciones), tot_features))

    for i in range(len(window_no_activaciones)):
        inicio = window_no_activaciones[i][0]
        fin = window_no_activaciones[i][1]

        x_no_act[i, -3] = num_casa
        x_no_act[i, -2] = inicio.value
        x_no_act[i, -1] = fin.value
        if len(((aggregate[inicio:fin]).to_numpy())[:, 0]) <= tot_features:
            x_no_act[i, :-3] = ((aggregate[inicio:fin]).to_numpy())[:, 0]
        else:
            x_no_act[i, :-3] = ((aggregate[inicio:fin]).to_numpy())[
                0:tot_features, 0
            ]  # elimino casos que pueden quedar de tamano 1 mayor

    return x_no_act, y_no_act


def eliminar_nan_y_filas_vacias(X, Y):
    #Se borran nan y filas de 0
    X = np.nan_to_num(X)
    Y = np.nan_to_num(Y)
    
    to_keep = np.sum(X, axis=1) != 0
    X = X[to_keep]
    Y = Y[to_keep]
    
    return X, Y

def matrices_de_activaciones(
    casas: list,
    electrodomestico: str,
    variante: str,
    parametros_activaciones: dict,
    win_len: timedelta,
    columna_referencia="aggregate",
    seed:int=None,
    verbose = False,
):
    """Dada una lista de casas y un electrodomestico devuelve las X e Y de ese electrodomestico. La variante define el vector de target
    si es autoencoder o rectangulos. El dict de parametros activaciones 
        
        Donde los ultimos 3 valores de X devueltos
        [-3] = num_casa
        [-2] = inicio.value
        [-1] = fin.value
    
    """
    if not seed is None:np.random.seed(seed)
    if verbose:
        print('--------------------------------------------------------')
        print(f'Calculo de activaciones para: {electrodomestico}\n')
    min_off = parametros_activaciones.get("min_off_duration")
    min_on = parametros_activaciones.get("min_on_duration")
    threshold = parametros_activaciones.get("on_power_threshold")
    tot_features = int(win_len.total_seconds() / 6) + 3
    X_act = np.empty([0, tot_features])
    X_no_act = np.empty([0, tot_features])
    if variante == "autoencoder":
        Y_act = np.empty([0, tot_features - 3])
        Y_no_act = np.empty([0, tot_features - 3])
    else:
        Y_act = np.empty([0, 3])
        Y_no_act = np.empty([0, 3])
    for casa in casas:
        if verbose:
            num_casa=casa["num_casa"]
            #print(f"Calculando Activaciones del {electrodomestico} en la Casa: {num_casa}")
        
        activaciones = activation_series_for_chunk_h5(
            casa[electrodomestico],
            min_off_duration=min_off,
            min_on_duration=min_on,
            on_power_threshold=threshold,
        )
        activaciones = activaciones[1:-1]
        if variante == "autoencoder":
            #if verbose:print("Calculando vectores X, y con activaciones para autoencoder")

            x_act, y_act = get_X_y_ae(
                activaciones,
                casa[columna_referencia],
                casa[electrodomestico],
                win_len=win_len,
                num_casa=casa["num_casa"],
            )

            #if verbose:print("Calculando vectores X, y sin activaciones para autoencoder")
            x_no_act, y_no_act = get_X_y_no_act_ae(
                activaciones,
                casa[columna_referencia],
                casa[electrodomestico],
                win_len,
                num_casa=casa["num_casa"],
            )
        else:
            #if verbose:print("Calculando vectores X, y con activaciones para rectangulos")
            x_act, y_act = get_X_y(
                activaciones,
                casa[columna_referencia],
                win_len=win_len,
                num_casa=casa["num_casa"],
            )
            #if verbose:print("Calculando vectores X, y sin activaciones para rectangulos")
            x_no_act, y_no_act = get_X_y_no_act(
                activaciones,
                casa[columna_referencia],
                win_len,
                num_casa=casa["num_casa"],
            )

        if verbose:
            #print("Stackeando X, y")
            print(f"Shape x_act \tpara la casa {num_casa}\t: {x_act.shape}")
            print(f"Shape x_no_act \tpara la casa {num_casa}\t: {x_no_act.shape}\n")
        X_act = np.vstack((X_act, x_act))
        Y_act = np.vstack((Y_act, y_act))
        X_no_act = np.vstack((X_no_act, x_no_act))
        Y_no_act = np.vstack((Y_no_act, y_no_act))


    if verbose:
        print(f"\nShape total X_act\t\t\t\t: {X_act.shape}")
        print(f"Shape total X_no_act\t\t\t\t: {X_no_act.shape}")
        print('--------------------------------------------------------')
        #print("\n____________________Fin____________________")
    return X_act, Y_act, X_no_act, Y_no_act
        