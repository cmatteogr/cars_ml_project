from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import json
import os

from constants import ARTIFACTS_FOLDER_PATH


def test(model_filename, X, y):
    """
    Test regression model to predict cars price
    :param model_filename: model Filename
    :param X: evaluation features
    :param y: evaluation target
    :return: Model testing results
    """
    print("Test Forward Neural Network - PyTorch model")

    # Load the model
    model = torch.load(str(os.path.join(ARTIFACTS_FOLDER_PATH, model_filename)))
    model.eval()
    # Transform input to tensor
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    # Predict evaluation dataset
    y_pred = model(X_tensor)
    y_pred_narray = y_pred.detach().numpy()

    # Calculate scores
    r2 = r2_score(y, y_pred_narray)
    mse = mean_squared_error(y, y_pred_narray)
    mae = mean_absolute_error(y, y_pred_narray)
    results_json = {
        'r2': r2,
        'mse': mse,
        'mae': mae,
    }
    print("results:", results_json)
    model_results_filepath = r'./data/evaluation/neural_network_pytorch_model_cars_price_prediction_results.json'
    with open(model_results_filepath, 'w') as f:
        json.dump(results_json, f)

    print("Test Forward Neural Network - PyTorch Completed")

    # Return model results
    return model_results_filepath
