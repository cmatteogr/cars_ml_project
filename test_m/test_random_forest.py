from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import json
import os

from constants import ARTIFACTS_FOLDER_PATH

def test(model_filename, X, y):
    """
    Test regression model to predict cars price
    :param model_filename: model filename
    :param X: test features
    :param y: test target
    :return: Model testing results
    """
    print("Train Random Forest Regressor model")

    # Load the model
    with open(os.path.join(ARTIFACTS_FOLDER_PATH, model_filename), 'rb') as file:
        regression_cars_price_model: RandomForestRegressor = pickle.load(file)
    # Predict test dataset
    y_pred = regression_cars_price_model.predict(X.values)

    # Calculate scores
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    results_json = {
        'r2': r2,
        'mse': mse,
        'mae': mae,
    }
    print("results:", results_json)
    model_results_filepath = r'./data/test/random_forest_model_cars_price_prediction_results.json'
    with open(model_results_filepath, 'w') as f:
        json.dump(results_json, f)

    print("Test Random Forest Regressor Completed")

    # Return model results
    return model_results_filepath
