from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Input
from tensorflow.keras.layers import Dense


def train(X, y):
    """
    Train regression model to predict cars price - Using Neural Networks

    :param X_train: training dataset
    :param y_train: training target

    :return: Model trained and results
    """

    model = Sequential()
    model.add(Input(shape=X.shape[1]))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=30, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=1))

    model.compile(loss='mse', optimizer='rmsprop')

    model.fit(X, y, epochs=100, verbose=0, validation_split=0.1, callbacks=None)

    model.evaluate(X, y, verbose=0)

    y_predict = model.predict(X, verbose=0)

    def build_model(hp):
        model = keras.Sequential()
        model.add(keras.layers.Dense(units=hp.Choice('units_layer1', [8, 16, 32]),activation=hp.Choice('activation_layer1', ['tanh', 'relu'])))
        model.add(keras.layers.Dense(units=hp.get('units_layer1'), activation='relu'))
        model.add(keras.layers.Dense(1, activation='relu'))
        model.compile(loss='mse')
        return model


    print("Train AutoML Regressor model")
    # Join the train set and train target
    X['price'] = y

    # Setup model
    clf = setup(X, target='price')

    # Comparing models to select the best one
    best_model = compare_models()

    # Creating a model - let's say a Random Forest Classifier
    # You can replace 'rf' with a model of your choice
    model = create_model('rf')

    # Optional: Tuning the model for better performance
    tuned_model = tune_model(model)

    # Finalizing the model (trains on the whole dataset)
    final_model = finalize_model(tuned_model)

    # Apply predictions
    predictions_df = predict_model(final_model, data=X)

    # Calculate scores
    r2 = r2_score(y, predictions_df['prediction_label'])
    mse = mean_squared_error(y, predictions_df['prediction_label'])
    mae = mean_absolute_error(y, predictions_df['prediction_label'])
    results_json = {
        'r2': r2,
        'mse': mse,
        'mae': mae,
    }
    print("results:",results_json)

    # Save model
    model_filepath = r'./data/train/automl_model_cars_price_prediction'
    save_model(final_model, model_filepath)

    # Save results
    train_filepath = r'./data/train/automl_cars_price_validation_prediction.csv'
    predictions_df.to_csv(train_filepath, index=False)

    print("Training AutoML Regressor Completed")

    # Return trained model and model results
    return model_filepath, results_json
