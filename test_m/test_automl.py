from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pycaret.regression import load_model


def test(model_filepath, X, y):
    """
    Test regression model to predict cars price
    :param model_filepath: training dataset
    :param X: test features
    :param y: test target
    :return: Model testing results
    """
    print("Test AutoML Regressor model")

    # Read the model
    regression_cars_price_model = load_model(model_filepath)
    # Predict test dataset
    predictions_df = regression_cars_price_model.predict(X)
    # Join the train set and train target

    # Calculate scores
    r2 = r2_score(y, predictions_df['prediction_label'])
    mse = mean_squared_error(y, predictions_df['prediction_label'])
    mae = mean_absolute_error(y, predictions_df['prediction_label'])
    results_json = {
        'r2': r2,
        'mse': mse,
        'mae': mae,
    }
    print("results:", results_json)

    print("Test AutoML Regressor Completed")

    # Return model results
    return results_json
