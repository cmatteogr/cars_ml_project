from preprocess.preprocess import preprocess
from train.train_random_forest import train as train_random_forest
from train.train_cat_boost import train as train_cat_boost
from train.train_automl import train as train_automl
from train.train_neural_network_tensorflow import train as train_neural_network_tensorflow
from train.train_neural_network_pytorch import train as train_neural_network_pytorch
from test_m.test_automl import test as test_automl
from test_m.test_cat_boost import test as test_cat_boost
from test_m.test_neural_network_pytorch import test as test_neural_network_pytorch
from test_m.test_neural_network_tensorflow import test as test_neural_network_tensorflow
from test_m.test_random_forest import test as test_random_forest
import json
import os
import shutil

from constants import ARTIFACTS_FOLDER_PATH, DEPLOYMENT_FOLDER_PATH

# Init datasource filepath
cars_filepath = r'.\data\data_exploration\input\cars.csv'
# Init model tool
model_tool = 'randomforest'

# If Neural Networks is used the normalize the data
scale_data = model_tool in ['neural_network_tensorflow', 'neural_network_pytorch']
train_inputer = False

# Preprocess
X_train, y_train, X_test, y_test, preprocess_config_data = preprocess(cars_filepath, test_size=0.15,
                                                                      train_inputer=train_inputer,
                                                                      scale_data=scale_data)

# Training
match model_tool:
    case 'randomforest':
        regression_model_filename, train_results_filename = train_random_forest(X_train, y_train)
    case 'catboost':
        regression_model_filename, train_results_filename = train_cat_boost(X_train, y_train)
    case 'automl':
        regression_model_filename, train_results_filename = train_automl(X_train, y_train)
    case 'neural_network_tensorflow':
        regression_model_filename, train_results_filename = train_neural_network_tensorflow(X_train, y_train)
    case 'neural_network_pytorch':
        regression_model_filename, train_results_filename = train_neural_network_pytorch(X_train, y_train)
    case _:
        raise Exception(f"Invalid training model: {model_tool}")

regression_model_filename = 'automl_model_cars_price_prediction'
# Test
match model_tool:
    case 'randomforest':
        model_results_filepath = test_random_forest(regression_model_filename, X_test, y_test)
    case 'catboost':
        model_results_filepath = test_cat_boost(regression_model_filename, X_test, y_test)
    case 'automl':
        model_results_filepath = test_automl(regression_model_filename, X_test, y_test)
    case 'neural_network_tensorflow':
        model_results_filepath = test_neural_network_tensorflow(regression_model_filename, X_test, y_test)
    case 'neural_network_pytorch':
        model_results_filepath = test_neural_network_pytorch(regression_model_filename, X_test, y_test)
    case _:
        raise Exception(f"Invalid Testing model: {model_tool}")

# Check if model can be deployed
r2_threshold = 0.90
print(f'Check trained model performance')
# Load testing results
with open(model_results_filepath, 'r') as json_file:
    test_model_results = json.load(json_file)
if test_model_results['r2'] < r2_threshold:
    raise Exception(
        f"R2 score below threshold is {r2_threshold}. The model doesn't have a good quality. Improve it: Current R2 {test_model_results['r2']}")

print(f'Trained model R2 score above: {r2_threshold}')
# Check if the model report data exist - previous model deployed
model_deployed_data_filename = 'cars_regressor_price_model_data.json'
if os.path.exists(os.path.join(ARTIFACTS_FOLDER_PATH, model_deployed_data_filename)):
    # If file exists then check if trained model has better performance
    with open(os.path.join(ARTIFACTS_FOLDER_PATH, model_deployed_data_filename), 'r') as json_file:
        current_model_data = json.load(json_file)
    # Compare R2
    if current_model_data['r2'] > test_model_results['r2']:
        raise Exception(f"R2 score below current model deployed is {test_model_results['r2']}")
    else:
        print(f"Trained model R2 score above Current model deployed is {test_model_results['r2']}")

# Deployment
print('Deploy Cars price prediction model')

# Build model data
model_data = {
    'r2': test_model_results['r2'],
    'model_tool': model_tool,
    'preprocess_config_data': preprocess_config_data,
    'model_price_regression_filepath': regression_model_filename,
}
with open(os.path.join(ARTIFACTS_FOLDER_PATH, model_deployed_data_filename), 'w') as f:
    json.dump(model_data, f)

# Save the models in the deployment projects
# Move the preprocess models to deploy
for key, m_filename in preprocess_config_data['models_filenames'].items():
    shutil.copy(str(os.path.join(ARTIFACTS_FOLDER_PATH, m_filename)),
                str(os.path.join(DEPLOYMENT_FOLDER_PATH, m_filename)))

print('Train pipeline completed')
