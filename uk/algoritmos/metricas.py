# -*- coding: utf-8 -*-
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

np.warnings.filterwarnings('ignore')
power_th = {"kettle": 2000, "fridge": 50, "washing": 20, "microwave": 200, "dish": 10}

def cls_rates(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    tipo_red: str = "rectangulos", 
    th: int = 0,
    ventana_deslizante: bool = False, 
    electrodomestico: str = None
    ) -> Tuple[np.int, np.int, np.int, np.int]:
    """Funcion que calcula TP,TN,FP y FN a partir de los vectores reales y predecidos
       permite aplicar un threshold de potencia a los vectores predecidos para considerarla nula"""
    if not ventana_deslizante:
        if tipo_red == "rectangulos":
            p_real = y_real[:, 0].copy()
            p_pred = y_pred[:, 0].copy()
        elif tipo_red == "autoencoder":
            p_real = np.mean(y_real, axis=1)
            p_pred = np.mean(y_pred, axis=1)
        else:
            raise ValueError("red must be one of ['rectangulos', 'autoencoder']")
        p_pred[p_pred <= th] = 0
    else: #si es ventana deslizante
        th_desag = power_th.get(electrodomestico)
        p_real = np.max(y_real, axis=1)
        p_real[p_real <= th_desag] = 0
        p_pred = np.max(y_pred, axis=1)
        p_pred[p_pred <= th] = 0
    tp = np.sum([b != 0 for a, b in zip(p_real, p_pred) if a != 0])
    tn = np.sum([b == 0 for a, b in zip(p_real, p_pred) if a == 0])
    fp = np.sum([b != 0 for a, b in zip(p_real, p_pred) if a == 0])
    fn = np.sum([b == 0 for a, b in zip(p_real, p_pred) if a != 0])
    return tp, tn, fp, fn


def cls_metrics(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    tipo_red: str ="rectangulos",
    th: int = 0,
    verbose: bool = True,
    ventana_deslizante: bool = False,
    elec: str = None,
    ) -> Tuple[np.int, np.int, np.int, np.int, np.int]:
    """Funcion que calcula recall, precision, accuracy y f1-score a partir de los vectores reales y predecidos
       permite aplicar un threshold de potencia a los vectores predecidos para considerarla nula"""
    tp, tn, fp, fn = cls_rates(y_real, y_pred, tipo_red, th, ventana_deslizante, elec)
    recall = tp / (tp + fn)        
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / np.shape(y_real)[0]
    fpr = fp / (tn + fp)
    #import ipdb; ipdb.set_trace()
    
    #print(f"th:{th},\t fp: {fp},\t tn: {tn},\t fp: {fp},\t fpr: {fpr}")
    f1 = 2 * (precision * recall) / (precision + recall)
    if verbose:
        print(f"Recall: {recall}")
        print("--------------------")
        print(f"Precision: {precision}")
        print("--------------------")
        print(f"Accuracy: {accuracy}")
        print("--------------------")
        print(f"False positive rate: {fpr}")
        print("--------------------")
        print(f"f1-score: {f1}")
    return recall, precision, accuracy, fpr, f1


def roc(real: np.ndarray,
        pred: np.ndarray,
        tipo_red: str = None, 
        ventana_deslizante: bool = False,
        puntos: int = None,
        elec:str = None
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                  np.ndarray, np.ndarray]:
    """Calculo de roc"""
    if puntos is None:
        puntos = 1000
    
    threshs = np.logspace(-5, 4, puntos).tolist()
    threshs.insert(0, -np.inf)
    threshs.insert(len(threshs), np.inf)
    threshs = np.array(threshs)
    
    #threshs = np.linspace(0, 4, 1000)
    # threshs = [0, 100, 10000]
    values = []
    for thresh in threshs:
        values.append(
            cls_metrics(
                real,
                pred,
                th=thresh,
                tipo_red=tipo_red,
                verbose=False,
                ventana_deslizante=ventana_deslizante,
                elec=elec,
            )
        )
    recalls = np.array([value[0] for value in values])
    precisions = np.array([value[1] for value in values])
    accuracys = np.array([value[2] for value in values])
    fprs = [value[3] for value in values]
    f1s = np.array([value[4] for value in values])
    
    return recalls, precisions, accuracys, fprs, f1s, threshs

def plot_roc(recalls: np.ndarray,  
             fprs: np.ndarray, 
             elec: str = None,
             argmax: int = None,
             umbral: int = None) -> None:
    # Plot ROC
    plt.figure()
    plt.plot(fprs, recalls)
    plt.title(f"ROC {elec}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")    
    if argmax:
        label = 'Punto de trabajo'
        if umbral:
            label += f'''\numbral = {umbral:.1f} Watts'''
        plt.plot(fprs[argmax],recalls[argmax], 'o', c='red', 
                 label=label)
    plt.grid()
    plt.legend()
    plt.show()

def auc(recalls:np.ndarray,  fprs: np.ndarray) -> int:
    try:
        auc = metrics.auc(fprs, recalls)
    except:
        auc = 0.5
    return auc



def MAE(y_real, y_pred, red="rectangulos"):
    if red == "rectangulos":
        mae = 0
        for i in range(y_real.shape[0]):
            N = y_real.shape[0]
            k = np.argmin([y_real[i, 1], y_pred[i, 1]])
            A = [y_real[i], y_pred[i]][k]
            B = [y_pred[i], y_real[i]][k]
            wa, p1a, p2a = A[0], A[1], A[2]
            wb, p1b, p2b = B[0], B[1], B[2]
            if p2a < p1b:
                mae += (wa * (p2a - p1a) + wb * (p2b - p1b)) / N
            elif p2a < p2b:
                mae += (
                    wa * (p1b - p1a) + np.abs(wa - wb) * (p2a - p1b) + wb * (p2b - p2a)
                ) / N
            elif p2b < p2a:
                mae += (
                    wa * (p1b - p1a) + np.abs(wa - wb) * (p2b - p1b) + wa * (p2a - p2b)
                ) / N
        return mae

    elif red == "autoencoder" or red == 'ventanas':
        mae = np.mean(np.abs(y_real - y_pred))
        return mae

    else:
        raise AttributeError

def potencia_media(y):
    # Lo que sigue es la potencia media en la ventana
    p = y[:, 0] * (y[:, 2] - y[:, 1])
    return p

def REITE(y_real, y_pred, red="rectangulos"):

    if red == "rectangulos":
        p_real = potencia_media(y_real)
        p_pred = potencia_media(y_pred)
        numerador = np.abs(np.sum(p_real) - np.sum(p_pred))
        denominador = np.max((np.sum(p_real), np.sum(p_pred)))
        reite = numerador / denominador
        return reite

    elif red == "autoencoder" or red == 'ventanas':
        p_real = np.mean(y_real)
        p_pred = np.mean(y_pred)
        numerador = np.abs(np.sum(p_real) - np.sum(p_pred))
        denominador = np.max((np.sum(p_real), np.sum(p_pred)))
        reite = numerador / denominador

        return reite
    else:
        raise AttributeError

    



