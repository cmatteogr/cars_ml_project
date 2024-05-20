from preprocess.preprocess import preprocess


#r'C:\Users\cesar_0qb0xal\Documents\GitHub\cars_ml_project\data\data_exploration\input\cars.csv'
cars_filepath = r'.\data\data_exploration\input\cars.csv'
X_train, y_train, X_test, y_test, imputer_model, outlier_removal_model = preprocess(cars_filepath, test_size=0.15, train_inputer=True)