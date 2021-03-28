# NILM-UY Dataset




# Algoritmos
Algoritmos de procesamiento y desagregacion de datos de la base UK-DALE.<br>
Los algoritmos se basan en [Neuronal NILM (Jack Kelly)](docs/neural_nilm.pdf). <br>
La explicacion de la implementacion especifica se puede ver en 
[Proyecto de fin de carrera (Marchsesoni-MariÃ±o-Masquil)](docs/MMM20.pdf)

Cuenta con varias funciones para la lectura y procesamiento de el arhivo .h5 suministrado en la base de datos UK-Dale.
Ademas, tiene algoritmos para entrenar y evaluar redes neuronales profundas.


## Procesamiento de datos
En el [notebook de procesamiento de datos](Generacion_X_y.ipynb) se puede ver como se utilizan las funciones para obtener datos utiles para entrenar redes neuronales.

## Algoritmos
El proceso de entrenamiento y evaluación se divide en 3 notebooks.

El primero es: [Notebook de entrenamiento de redes neuronales](EntrenamientoRedesNeuronales.ipynb). Este contiene codigo capaz de entrenar una arquitectura
de re neuronal para un electrodomestico. Además se guardan los pesos generados por el entrenamiento y los valores de evolucion de la loss de entrenamiento
Adicionamente se tiene un script capaz de  entrenar todos los modelos para todos los electrodomesticos.

El segundo es: [Notebook de metricas](MetricasRedesNeuronales.ipynb). Este carga el modelo entrenado en el primer notebook, predice los valores para cada conjunto de datos
y reporta las metricas. Estas son: AUC, Recall, Precision, Accuracy, False Positive Rate, F1-Score, Reite, MAE

El tercero es: [Notebook de evaluacion por metodo de ventanas deslizantes](VentanasDeslizantes.ipynb). Este carga el modelo entrenado en el primer notebook,
y la serie temporal completa. Va prediciendo la salida por ventanas (una ventana que se mueve). Reporta las metricas para la serie completa.

## Datos disponibles
[ukdale.h5](https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated/ukdale.h5.zip)
<br>[data_ini.pickle](https://iie.fing.edu.uy/~cmarino/NILM/datos_ini.pickle)
<br>[data_fin.pickle](https://iie.fing.edu.uy/~cmarino/NILM/datos_fin.pickle)
<br>[datos_uruguay.pickle](https://iie.fing.edu.uy/~cmarino/NILM/datos_uruguay.pickle)
<br>[vectores.zip](https://iie.fing.edu.uy/~cmarino/NILM/vectores.zip)
<br>[pesos.zip](https://iie.fing.edu.uy/~cmarino/NILM/pesos.zip)

Link alternativo a Google Drive:<br>
[Datos](https://drive.google.com/drive/folders/1AOkR5vRICbf8NUeMc40w3UYwXxuqjnr-?usp=sharing) 

Nota: Los datos fueron creados con seed=5.
