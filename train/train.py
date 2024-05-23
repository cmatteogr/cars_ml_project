
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.ensemble import RandomForestRegressor


def train(X_train, y_train):
    """
    
    """
    # Define the parameter bayesian to search over
    param_space = {
        'n_estimators': Integer(10, 100),  # Number of trees in the forest
        'max_depth': Integer(1, 50),       # Maximum depth of the tree
        'min_samples_split': Real(0.01, 1.0, 'uniform'),  # Minimum number of samples required to split an internal node
    }

    # Initialize Random Forest Regressor algorithm 
    regression_model = RandomForestRegressor(random_state=42, criterion='squared_error')

    optimizer = BayesSearchCV(
        estimator=regression_model,
        search_spaces=param_space,
        n_iter=25,  # Number of parameter settings that are sampled
        cv=5,       # 5-fold cross-validation
        random_state=42
    )

    # Fit the BayesSearchCV to the training data
    optimizer.fit(X_train, y_train)

    


    return regression_model, results_json
