# NILM-UY Dataset
This repository contains a processed sample of the NILM-UY dataset and an implementation of the algorithms proposed in *NILM: Multivariate DNN performance analysis with high frequency features*. Instructions on how to get access to the full raw data is also included.


# Code
* Pre-processing algorithms for the UK-Dale dataset.
<br>
These algorithms are based on [Neural NILM (Jack Kelly)](docs/neural_nilm.pdf). These functions are useful for reading and pre-processing the `h5` file from UK-Dale.
* Training and evaluation scripts for the models proposed in the paper.


## Data pre-processing
The [pre-processing notebook](Generacion_X_y.ipynb) serves as an example on how to process the UK-Dale dataset for training the dissagregation models.

## Algorithms
The training and evaluation procedure is divided into three notebooks.

1. [Training notebook](EntrenamientoRedesNeuronales.ipynb).
This notebook contains the code for training a single model for one appliance. We also included an script for training all the architectures for all the considered appliances.

2. [Metrics notebook](MetricasRedesNeuronales.ipynb). 
This notebook allows you to load a previously trained model and calculates the metrics reported in the paper. AUC, Recall, Precision, Accuracy, False Positive Rate, F1-Score, Reite, MAE.

3. [Rolling windows evaluation](VentanasDeslizantes.ipynb). 
This notebook can be used for evaluating the previously trained models with the rolling windows approach. Contains the code for loading the whole power time series and making predictions in a rolling window fashion.

# Data

* Pre-processed UK-Dale dataset ready for being used with the code base provided in this project. The dataset is already temporarily splitted: [data_ini.pickle](https://iie.fing.edu.uy/~cmarino/NILM/datos_ini.pickle) and
[data_fin.pickle](https://iie.fing.edu.uy/~cmarino/NILM/datos_fin.pickle)
* Pre-processed NILM-UY dataset. The uruguayan dataset already pre-processed with a sampling period of 6 seconds. This data is also ready for being used with the code provided in this project. [datos_uruguay.pickle](https://iie.fing.edu.uy/~cmarino/NILM/datos_uruguay.pickle)
* Trained weights of the models [pesos.zip](https://iie.fing.edu.uy/~cmarino/NILM/pesos.zip)

Alternative Google Drive link:<br>
[Data](https://drive.google.com/drive/folders/1AOkR5vRICbf8NUeMc40w3UYwXxuqjnr-?usp=sharing) 

## How to get full acess to the NILM-UY dataset
The raw NILM-UY dataset collected in Uruguay contains aggregated and disaggregated data:
*  High sampling frequency aggregated data from two homes. (140 gb)
*  Individual power measurements per appliance. With a sampling period of 1 minute. (37 mb)

If you are interested in accessing this data for research purposes send us an email at `cmarino@fing.edu.uy` or `emasquil@fing.edu.uy` and we can provide you with download links.