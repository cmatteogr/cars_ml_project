from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import json
import os

from constants import ARTIFACTS_FOLDER_PATH


def train(X, y):
    """
    Train regression model to predict cars price - Using Neural Networks with TensorFlow
    :param X: training dataset
    :param y: training target
    :return: Model trained and results
    """
    print("Train Forward Neural Network - TensorFlow model")

    # Define the model
    model = Sequential()
    model.add(Input(shape=(X.shape[1],)))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=30, activation='relu'))
    model.add(Dense(units=15, activation='relu'))
    model.add(Dense(units=1))

    # Define compilation method, loss function
    model.compile(loss='mse', optimizer='rmsprop')

    # Train the model
    n_epochs = 1000  # number of epochs to run
    validation_split = 0.2  # validation percentage size
    batch_size = 32  # size of each batch
    # init the early stopping callback
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    # Train model
    history = model.fit(X, y, epochs=n_epochs, verbose=1, batch_size=batch_size, validation_split=validation_split,
                        callbacks=[callback])
    # Evaluate model
    model.evaluate(X, y, verbose=0)

    # Calculate scores
    y_pred = model.predict(X, verbose=0)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    results_json = {
        'r2': r2,
        'mse': mse,
        'mae': mae,
    }
    print("results:", results_json)

    # Plot the history training
    hist = history.history
    plt.plot(hist['loss'])
    model_history_filepath = r'./data/train/neural_network_tensorflow_model_history_training.jpg'
    plt.savefig(model_history_filepath)

    # Build predictions report
    # Copy Input dataset to generate report
    predictions_df = X.copy()
    predictions_df['price'] = y
    predictions_df['prediction_price'] = y_pred

    # Save results
    train_filepath = r'./data/train/neural_network_tensorflow_cars_price_validation_prediction.csv'
    predictions_df.to_csv(train_filepath, index=False)

    # Save model and results
    model_filename = 'neural_network_tensorflow_model_cars_price_prediction.keras'
    model.save(os.path.join(ARTIFACTS_FOLDER_PATH, model_filename))
    model_results_filename = 'neural_network_tensorflow_model_cars_price_prediction_results.json'
    with open(os.path.join(ARTIFACTS_FOLDER_PATH, model_results_filename), 'w') as f:
        json.dump(results_json, f)

    print("Training Forward Neural Network - TensorFlow Completed")

    # Return trained model and model results
    return model_filename, model_results_filename
