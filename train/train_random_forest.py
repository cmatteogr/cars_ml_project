from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle
import json
import os

from constants import ARTIFACTS_FOLDER_PATH


def train(X, y):
    """
    Train regression model to predict cars price
    :param X: training dataset
    :param y: training target
    :return: Model trained and results
    """
    print("Train Random Dores Regressor model")
    # Define the parameter bayesian to search over
    param_space = {
        'n_estimators': Integer(10, 100),  # Number of trees in the forest
        'max_depth': Integer(1, 100),  # Maximum depth of the tree
        'min_samples_split': Real(0.01, 1.0, 'uniform'),  # Minimum number of samples required to split an internal node
    }

    # Split train and Validation dataset
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.15, random_state=42)

    # Initialize Random Forest Regressor algorithm 
    regression_model = RandomForestRegressor(random_state=42, criterion='squared_error')
    # Search the best hyperparameters
    optimizer = BayesSearchCV(
        estimator=regression_model,
        search_spaces=param_space,
        n_iter=25,  # Number of parameter settings that are sampled
        cv=5,  # 5-fold cross-validation
        random_state=42,
        verbose=1
    )

    # Fit the Prediction model to the training data
    optimizer.fit(X_train.values, y_train.values)

    # Get best model
    best_estimator = optimizer.best_estimator_
    print(f'Best Estimator: {best_estimator}')

    # Get the best parameters
    best_params = optimizer.best_params_
    print(f'Best Parameters: {best_params}')

    # Get the best score
    best_score = optimizer.best_score_
    print(f'Best CV Score: {best_score}')

    # Apply model predictions
    y_pred = best_estimator.predict(X_train.values)
    y_pred_validation = best_estimator.predict(X_validation.values)

    # Calculate scores
    r2 = r2_score(y_train, y_pred)
    mse = mean_squared_error(y_train, y_pred)
    mse_validation = mean_squared_error(y_validation, y_pred_validation)
    mae = mean_absolute_error(y_train, y_pred)
    results_json = {
        'r2': r2,
        'mse': mse,
        'mae': mae,
    }
    print("results:", results_json)

    # Check if model is overfitted
    if mse_validation > mse * 1.1:
        print(f'The model may be overfitting. train MSE: {mse}, validation MSE: {mse_validation}')
        raise Exception("The model may be overfitting")
    else:
        print(f'The model is not overfitting. train MSE: {mse}, validation MSE: {mse_validation}')

    # Save model and results
    model_filename = 'random_forest_model_cars_price_prediction.pkl'
    with open(os.path.join(ARTIFACTS_FOLDER_PATH, model_filename), 'wb') as file:
        pickle.dump(best_estimator, file)
    model_results_filename = 'random_forest_model_cars_price_prediction_results.json'
    with open(os.path.join(ARTIFACTS_FOLDER_PATH, model_results_filename), 'w') as json_file:
        json.dump(results_json, json_file)

    # Save model results
    X_train['price'] = y_train
    X_validation['price'] = y_validation
    X_train['prediction_price'] = y_pred
    X_validation['prediction_price'] = y_pred_validation
    train_filepath = './data/train/random_forest_cars_price_train_prediction.csv'
    X_train.to_csv(train_filepath, index=False)
    validation_filepath = './data/train/random_forest_cars_price_validation_prediction.csv'
    X_train.to_csv(validation_filepath, index=False)

    print("Training Random Fores Regressor Completed")

    # Return trained model and model results
    return model_filename, model_results_filename
