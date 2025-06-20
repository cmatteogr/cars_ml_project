from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import json
import os

from constants import ARTIFACTS_FOLDER_PATH


def test(model_filename, X, y):
    """
    Test regression model to predict cars price
    :param model_filename: model filename
    :param X: evaluation features
    :param y: evaluation target
    :return: Model testing results
    """
    print("Test Forward Neural Network - PyTorch model")

    # Load the model
    model = tf.keras.models.load_model(os.path.join(ARTIFACTS_FOLDER_PATH, model_filename))
    # Predict evaluation dataset
    y_pred = model.predict(X, verbose=0)

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
    model_results_filepath = r'./data/evaluation/neural_network_tensorflow_model_cars_price_prediction_results.json'
    with open(model_results_filepath, 'w') as f:
        json.dump(results_json, f)

    print("Test Forward Neural Network - PyTorch Completed")

    # Return model results
    return model_results_filepath
