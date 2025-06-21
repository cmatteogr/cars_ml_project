# Cars Price Prediction - US Market

![Should Dollar Rate be an Excuse for Car Price Increase_ NO!!!](https://github.com/cmatteogr/cars_ml_project/assets/138587358/0e804a48-f2b5-4c77-a12a-16163f1244c1)

Do you want to predict the Cars prices from US? This project will help you.
Cars Price Prediction project uses the data collected from [Cars Scrapy](https://github.com/cmatteogr/cars_scrapy) and use it to predict the Cars prices based on their features.

This project has been used in the Medellín Machine Learning - Study Group (MML-SG) to understand the Machine Learning bases. Therefore, the project was built end-to-end from scratch

You will find in this repo:
* Data Exploration using Jupyter to understand the data, its quality and define relevant features and their transformations.
* Model Training Pipeline:
  - Preprocess script used to apply validations, filtering, transformations, inputations, outliers removal, and normalization in the dataset.
  - Training script used to train multiple regression models (Random Forest, CatBoost, AutoML, Neural Networs) all of them with the same goal: predict cars prices highlighting the advantages and disadvantages and the setup needed to train them
  - Testing script used to evaluate the model performance using dedicated metrics like R2, RMSE, MSE, MAE and residuals plot.
  - Deployment conditionals is used to define when a model can be deployed based on business rules and/or model performance to complete the model training pipeline.
 
## Prerequisites
* Install Python 3.11
* Install the libraries using requirements.txt.
```bash
pip install -r requirements.txt
```
* Add the cars.csv dataset CSV file (Check [Cars Scrapy](https://github.com/cmatteogr/cars_scrapy) project) in the folder .\data\data_exploration\input\

## Usage
Execute the script train_pipeline_cars_price_regressor.py to start the model training pipeline, you can select the model to train using the variable 'model_tool' (e.g. model_tool = 'randomforest'), the preprocess script works the same for all the models except for the neural networks models where Normalization transformation is executed
temp_inference.py
```bash
python train_pipeline_cars_price_regressor.py
```
**NOTE**: Depending on the model to train the resources/time needed change so be patient or be sure you are using appropriate CPU-GPU instance.

## External Resoruces
This project was built by the Medellín Machine Learning - Study Group (MML-SG) community. In the following [link](https://drive.google.com/drive/u/0/folders/1nPMtg6caIef5o9S_J8WyNEvyEt5sO1VH) you can find the meetings records about it:
* [2. Exploración de Modelos de ML y Exploración de Datos (2024-02-28 19:14 GMT-5)](https://drive.google.com/file/d/1mqpccGVjhOQTDV5c80RKk1ECNnK6DCqn/view?usp=drive_link)
* [3. Análisis de Datos y Selección de Variables para Modelado (2024-03-06 19:08 GMT-5)](https://drive.google.com/file/d/1N9LrEJ3TYRZY6Fumxor3HircIahtwM24/view?usp=drive_link)
* [4. Construcción del Modelo de Predicción - Supervised Learning(2024-03-13 19:07 GMT-5)](https://drive.google.com/file/d/1PgFWmeBnIu__lHYkYQ4wIvJzyWro0tXM/view?usp=drive_link)
* [5. Supervised Learning - Optimización del Modelo (2024-04-10 19:11 GMT-5)](https://drive.google.com/file/d/1rIbYSJ5sGrCeNTGw74bfh6rVzWtXP2UJ/view?usp=drive_link)
* [6. Implementación de la Detección de Anomalías (2024-04-17 19:09 GMT-5)](https://drive.google.com/file/d/1NU6CLKnL_O4xxduqQlrtPCgiFCZQKiI4/view?usp=drive_link)
* [7. Evaluación del Modelo y Resultados de la Detección de Anomalías (2024-04-24 19:09 GMT-5)](https://drive.google.com/file/d/1IFQ1AFlBal3UAFbdfB474GRovBOQUXaw/view?usp=drive_link)
* [5. Supervised Learning - Optimización del Modelo (2024-04-10 19:11 GMT-5)](https://drive.google.com/file/d/1rIbYSJ5sGrCeNTGw74bfh6rVzWtXP2UJ/view?usp=drive_link)
* [10. Revisión Final y Lecciones Aprendidas del Proyecto de Predicción de Precios de Carros (2024-05-15 19:11 GMT-5)](https://drive.google.com/file/d/1N91o4rzD-mr61eRiLeKb_cgQ1MJHXGPt/view?usp=drive_link)
