
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle


def train(X, y):
    """
    Train regression model to predict cars price

    :param X_train: training dataset
    :param y_train: training target

    :return: Model trained and results
    """
    print("Train Cat Boost model")
    # Define the parameter bayesian to search over
    param_space = {
        'learning_rate': Real(0.01, 0.3),
        'depth': Integer(3, 5),
        'iterations': Integer(10, 100),
        'l2_leaf_reg': Real(3, 10)
    }

    # Split train and Validation dataset
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.15, random_state=42)

    # Initialize Random Forest Regressor algorithm 
    catboost_model = CatBoostRegressor(verbose=1, random_state=42)

    # Search the best hyperparameters
    optimizer = BayesSearchCV(
        estimator=catboost_model,
        search_spaces=param_space,
        n_iter=30,  # Number of parameter settings that are sampled
        cv=5,       # 4-fold cross-validation
        random_state=42
    )
    
    # Fit the BayesSearchCV to the training data
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

    feature_importances = best_estimator.get_feature_importance()
    print(f'Feature Importances: {feature_importances}')

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
    print("results:",results_json)

    # Check if model is overfittied
    if mse_validation > mse * 1.1:
        print(f'The model may be overfitting. train MSE: {mse}, validation MSE: {mse_validation}')
        raise Exception("The model may be overfitting")
    else:
        print(f'The model is not overfitting. train MSE: {mse}, validation MSE: {mse_validation}')

    # Save model
    model_filepath = './data/train/cat_boost_model_cars_price_prediction.pkl'
    with open(model_filepath, 'wb') as file:
        pickle.dump(best_estimator, file)
    
    X_train['price'] = y_train
    X_validation['price'] = y_validation
    X_train['prediction_price'] = y_pred
    X_validation['prediction_price'] = y_pred_validation
    train_filepath = './data/train/cat_boost_cars_price_train_prediction.csv'
    X_train.to_csv(train_filepath, index=False)
    validation_filepath = './data/train/cat_boost_cars_price_validation_prediction.csv'
    X_train.to_csv(validation_filepath, index=False)

    print("Training CatBoost Regressor Completed")

    # Return trained model and model results
    return model_filepath, results_json, train_filepath, validation_filepath
