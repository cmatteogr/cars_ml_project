from typing import Callable
from fastapi import FastAPI
from app.services.model import MLModel
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import IsolationForest
from gensim.models import Word2Vec
import os
import json
import pickle
import joblib


def _startup_model(app: FastAPI, model_path: str, models_names: tuple) -> None:
    # Read model data file
    model_data_filepath = os.path.join(model_path, 'cars_regressor_price_model_data.json')
    with open(model_data_filepath, 'r') as json_file:
        model_data = json.load(json_file)
    # Load preprocess models
    # Load drivetrain Encoder model
    ohe_drivetrain_model: OneHotEncoder = joblib.load(
        os.path.join(model_path, model_data['ohe_drivetrain_model_filename']))
    # Load make Encoder model
    ohe_make_model: OneHotEncoder = joblib.load(os.path.join(model_path, model_data['ohe_make_model_filename']))
    # Load Body style Encoder model
    ohe_bodystyle_model: OneHotEncoder = joblib.load(
        os.path.join(model_path, model_data['ohe_bodystyle_model_filename']))
    # Load Fuel type Encoder model
    ohe_fuel_type_model: OneHotEncoder = joblib.load(
        os.path.join(model_path, model_data['ohe_fuel_type_model_filename']))
    # Load Exterior color Encoder model
    w2v_exterior_color_model = Word2Vec.load(os.path.join(model_path, model_data['w2v_exterior_color_model_filename']))
    # Load Interior color Encoder model
    w2v_interior_color_model = Word2Vec.load(os.path.join(model_path, model_data['w2v_interior_color_model_filename']))
    # Load Cat Encoder model
    w2v_cat_model = Word2Vec.load(os.path.join(model_path, model_data['w2v_cat_model_filename']))
    # Load imputer model
    imputer_model: IterativeImputer = joblib.load(os.path.join(model_path, model_data['imputer_model_filename']))
    # Load outlier detector model
    outlier_detector_model: IsolationForest = joblib.load(
        os.path.join(model_path, model_data['anomaly_detection_model_filename']))
    # Load scaler model
    scaler_model: IsolationForest = joblib.load(os.path.join(model_path, model_data['scaler_model_filename']))

    # Init the ML object and add to dict
    model_filepath = os.path.join(model_path, model_data['regression_model_filename'])
    model_tool = model_data['model_tool']
    model_instance = MLModel(model_filepath, model_tool)

    model_dict = {
        'preprocess': {
            'ohe_drivetrain_model': ohe_drivetrain_model,
            'ohe_make_model': ohe_make_model,
            'ohe_bodystyle_model': ohe_bodystyle_model,
            'ohe_fuel_type_model': ohe_fuel_type_model,

        }
    }

    model_folders = [os.path.join(model_path, item) for item in items if os.path.isdir(os.path.join(model_path, item))]
    # Init model dict and model extensions
    model_dict = {}
    # For each folder generate load the models
    for model_folder in model_folders:
        # List all files in the directory
        model_folder_files = [os.path.join(model_folder, file) for file in os.listdir(model_folder)]

        model_files_dict = {}
        # For each model in folder create a MLModel object
        for model_name in models_names:
            try:
                # Filter the model file
                model_filepath = next(filter(lambda m: str(m).endswith(model_name), model_folder_files))
            except StopIteration:
                # If no elements are found
                raise Exception(print(f"No model file found {model_name} in folder {model_folder}."))
            # Init the ML object and add to dict
            model_instance = MLModel(model_filepath)
            model_files_dict[model_name] = model_instance

        # Add location models dict
        location_name = model_folder.split("\\")[-1]
        model_dict[location_name] = model_files_dict

    # Add to app state model
    app.state.model = model_dict


def _shutdown_model(app: FastAPI) -> None:
    app.state.model = None


def start_app_handler(app: FastAPI, model_path: str, models_names: tuple) -> Callable:
    def startup() -> None:
        _startup_model(app, model_path, models_names)

    return startup


def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        _shutdown_model(app)

    return shutdown
