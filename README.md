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
 
## Requirements
* Install Python 3.11
* Install the libraries using requirements.txt.
* Add the cars.csv dataset CSV file (Check [Cars Scrapy](https://github.com/cmatteogr/cars_scrapy) project) in the folder .\data\data_exploration\input\

## Usage
Execute the script train_pipeline_cars_price_regressor.py to start the model training pipeline, you can select the model to train using the variable 'model_tool' (e.g. model_tool = 'randomforest'), the preprocess script works the same for all the models except for the neural networks models where Normalization transformation is executed

**NOTE**: Depending on the model to train the resources/time needed change so be patient or be sure you are using appropriate CPU-GPU instance.
