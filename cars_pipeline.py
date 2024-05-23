from preprocess.preprocess import preprocess
from train.train import train



cars_filepath = r'.\data\data_exploration\input\cars.csv'

# Preprocess
X_train, y_train, X_test, y_test, imputer_model, outlier_removal_model = preprocess(cars_filepath, test_size=0.15, train_inputer=True)

# Train
regression_model, results_json = train(X_train, y_train)