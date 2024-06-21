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

# Init datasource filepath
cars_filepath = r'.\data\data_exploration\input\cars.csv'
# Init model tool
model_tool = 'neural_network_tensorflow'

# If Neural Networks is used the normalize the data
scale_data = model_tool in ['neural_network_tensorflow', 'neural_network_pytorch']
train_inputer = False

# Preprocess
preprocess_result = preprocess(cars_filepath, test_size=0.15, train_inputer=train_inputer, scale_data=scale_data)
# Get the preprocess result
X_train = preprocess_result[0]
y_train = preprocess_result[1]
X_test = preprocess_result[2]
y_test = preprocess_result[3]
imputer_model_filepath = preprocess_result[4]
outlier_removal_model_filepath = preprocess_result[5]
scale_model_filepath = preprocess_result[6]
ohe_drivetrain_model_filepath = preprocess_result[7]
ohe_make_model_filepath = preprocess_result[8]
ohe_bodystyle_model_filepath = preprocess_result[9]
ohe_fuel_type_model_filepath = preprocess_result[10]
w2v_exterior_color_model_filepath = preprocess_result[11]
w2v_interior_color_model_filepath = preprocess_result[12]
w2v_cat_model_filepath = preprocess_result[13]

# Training
match model_tool:
    case 'randomforest':
        regression_model_filepath, train_results_filepath = train_random_forest(X_train, y_train)
    case 'catboost':
        regression_model_filepath, train_results_filepath = train_cat_boost(X_train, y_train)
    case 'automl':
        regression_model_filepath, train_results_filepath = train_automl(X_train, y_train)
    case 'neural_network_tensorflow':
        regression_model_filepath, train_results_filepath = train_neural_network_tensorflow(X_train, y_train)
    case 'neural_network_pytorch':
        regression_model_filepath, train_results_filepath = train_neural_network_pytorch(X_train, y_train)
    case _:
        raise Exception(f"Invalid training model: {model_tool}")

# Test
match model_tool:
    case 'randomforest':
        model_results_filepath = test_random_forest(regression_model_filepath, X_test, y_test)
    case 'catboost':
        model_results_filepath = test_cat_boost(regression_model_filepath, X_test, y_test)
    case 'automl':
        model_results_filepath = test_automl(regression_model_filepath, X_test, y_test)
    case 'neural_network_tensorflow':
        model_results_filepath = test_neural_network_tensorflow(regression_model_filepath, X_test, y_test)
    case 'neural_network_pytorch':
        model_results_filepath = test_neural_network_pytorch(regression_model_filepath, X_test, y_test)
    case _:
        raise Exception(f"Invalid Testing model: {model_tool}")

# Check if model can be deployed
r2_threshold = 0.90
print(f'Check trained model performance')
# Load testing results
with open(model_results_filepath, 'r') as json_file:
    test_model_results = json.load(json_file)
if test_model_results['r2'] < r2_threshold:
    raise Exception(f"R2 score below threshold is {r2_threshold}. The model doesn't have a good quality. Improve it.")

print(f'Trained model R2 score above: {r2_threshold}')
# Check if the model report data exist - previous model deployed
model_deployed_data_filepath = './artifacts/cars_regressor_price_model_data.json'
if os.path.exists(model_deployed_data_filepath):
    # If file exists then check if trained model has better performance
    with open(model_deployed_data_filepath, 'r') as json_file:
        current_model_data = json.load(json_file)
    # Compare R2
    if current_model_data['r2'] < test_model_results['r2']:
        raise Exception(f"R2 score below current model deployed is {test_model_results['r2']}")
    else:
        print(f"Trained model R2 score above Current model deployed is {test_model_results['r2']}")

# Deployment
print('Deploy Cars price prediction model')

# Get files names
imputer_model_filename = os.path.basename(imputer_model_filepath)
outlier_removal_model_filename = os.path.basename(outlier_removal_model_filepath)
scale_model_filename = os.path.basename(scale_model_filepath)
regression_model_filename = os.path.basename(regression_model_filepath)
model_results_filename = os.path.basename(model_results_filepath)
model_deployed_data_filename = os.path.basename(model_deployed_data_filepath)
ohe_drivetrain_model_filename = os.path.basename(ohe_drivetrain_model_filepath)
ohe_make_model_filename = os.path.basename(ohe_make_model_filepath)
ohe_bodystyle_model_filename = os.path.basename(ohe_bodystyle_model_filepath)
ohe_fuel_type_model_filename = os.path.basename(ohe_fuel_type_model_filepath)
w2v_exterior_color_model_filename = os.path.basename(w2v_exterior_color_model_filepath)
w2v_interior_color_model_filename = os.path.basename(w2v_interior_color_model_filepath)
w2v_cat_model_filename = os.path.basename(w2v_cat_model_filepath)

# Build model data
model_data = {
    'r2': test_model_results['r2'],
    'model_tool': model_tool,
    'imputer_model_filename': imputer_model_filename,
    'anomaly_detection_model_filename': outlier_removal_model_filename,
    'scaler_model_filename': scale_model_filename,
    'regression_model_filename': regression_model_filename,
    'model_results_filename': model_results_filename,
    'model_deployed_data_filename': model_deployed_data_filename,
    'ohe_drivetrain_model_filename': ohe_drivetrain_model_filename,
    'ohe_make_model_filename': ohe_make_model_filename,
    'ohe_bodystyle_model_filename': ohe_bodystyle_model_filename,
    'ohe_fuel_type_model_filename': ohe_fuel_type_model_filename,
    'w2v_exterior_color_model_filename': w2v_exterior_color_model_filename,
    'w2v_interior_color_model_filename': w2v_interior_color_model_filename,
    'w2v_cat_model_filename': w2v_cat_model_filename,
}
with open(model_deployed_data_filepath, 'w') as f:
    json.dump(model_data, f)

# Save the models in the deployment projects
d_filepath = './app/ml_model'  # Deployment folder path
shutil.copy(imputer_model_filepath, os.path.join(d_filepath, imputer_model_filename))
shutil.copy(outlier_removal_model_filepath, os.path.join(d_filepath, outlier_removal_model_filename))
shutil.copy(scale_model_filepath, os.path.join(d_filepath, scale_model_filename))
shutil.copy(regression_model_filepath, os.path.join(d_filepath, regression_model_filename))
shutil.copy(model_results_filepath, os.path.join(d_filepath, model_results_filename))
shutil.copy(model_deployed_data_filepath, os.path.join(d_filepath, model_deployed_data_filename))
shutil.copy(ohe_drivetrain_model_filepath, os.path.join(d_filepath, ohe_drivetrain_model_filename))
shutil.copy(ohe_make_model_filepath, os.path.join(d_filepath, ohe_make_model_filename))
shutil.copy(ohe_bodystyle_model_filepath, os.path.join(d_filepath, ohe_bodystyle_model_filename))
shutil.copy(ohe_fuel_type_model_filepath, os.path.join(d_filepath, ohe_fuel_type_model_filename))
shutil.copy(w2v_exterior_color_model_filepath, os.path.join(d_filepath, w2v_exterior_color_model_filename))
shutil.copy(w2v_interior_color_model_filepath, os.path.join(d_filepath, w2v_interior_color_model_filename))
shutil.copy(w2v_cat_model_filepath, os.path.join(d_filepath, w2v_cat_model_filename))

print('Train pipeline completed')
