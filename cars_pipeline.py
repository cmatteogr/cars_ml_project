from preprocess.preprocess import preprocess
from train.train_random_forest import train as train_random_forest
from train.train_cat_boost import train as train_cat_boost
from train.train_automl import train as train_automl



cars_filepath = r'.\data\data_exploration\input\cars.csv'

# Preprocess
X_train, y_train, X_test, y_test, imputer_model, outlier_removal_model = preprocess(cars_filepath, test_size=0.15, train_inputer=False)

# Trainiop[] 
train_model = 'automl'
match train_model:
    case 'randomforest':
        regression_model, results_json = train_random_forest(X_train, y_train)
    case 'catboost':
        regression_model, results_json = train_cat_boost(X_train, y_train)
    case 'automl':
        regression_model, results_json = train_automl(X_train, y_train)
    case _:
        raise Exception(f"Invalid training model: {train_model}")