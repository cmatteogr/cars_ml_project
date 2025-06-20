from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pycaret.regression import *
import os
import json

from constants import ARTIFACTS_FOLDER_PATH


def test(model_filename, X, y):
    """
    Test regression model to predict cars price
    :param model_filename: model filename
    :param X: evaluation features
    :param y: evaluation target
    :return: Model testing results
    """
    print("Test AutoML Regressor model")

    # Read the model
    regression_cars_price_model = load_model(str(os.path.join(ARTIFACTS_FOLDER_PATH, model_filename)))
    # Predict evaluation dataset
    predictions_df = predict_model(regression_cars_price_model, data=X)
    # Join the train set and train target

    # Calculate scores
    y = y.sort_values()
    predictions = predictions_df['prediction_label'].sort_values()
    r2 = r2_score(y, predictions)
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    results_json = {
        'r2': r2,
        'mse': mse,
        'mae': mae,
    }
    print("results:", results_json)
    model_results_filepath = r'./data/evaluation/automl_model_cars_price_prediction_results.json'
    with open(model_results_filepath, 'w') as f:
        json.dump(results_json, f)

    print("Test AutoML Regressor Completed")

    # Return model results
    return model_results_filepath
