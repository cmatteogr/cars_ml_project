from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def train(X, y):
    """
    Train regression model to predict cars price - Using Neural Networks with PyTorch

    :param X_train: training dataset
    :param y_train: training target

    :return: Model trained and results
    """
    print("Train Forward Neural Network - PyTorch model")

    # train-test split for model evaluation
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8, shuffle=True)

    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
    X_validation = torch.tensor(X_validation.values, dtype=torch.float32)
    y_validation = torch.tensor(y_validation.values, dtype=torch.float32).reshape(-1, 1)

    # Define the model
    # Forward Neural Network dense connected,
    model_input_shape = X_train.shape[1:][0]  # Input model shape
    model = nn.Sequential(
        nn.Linear(model_input_shape, 50),
        nn.ReLU(),
        nn.Linear(50, 30),
        nn.ReLU(),
        nn.Linear(30, 15),
        nn.ReLU(),
        nn.Linear(15, 1)
        # Last layer doesn't have activation function to allow continuous output, without a limited range using linear activation (wiegths * inputs)
    )

    # loss function and optimizer
    loss_fn = nn.MSELoss()  # mean square error
    # Select the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Init training variables
    n_epochs = 200  # number of epochs to run
    batch_size = 32  # size of each batch
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Hold the best model
    best_mse = np.inf  # init to infinity
    best_weights = None
    history = []

    # Train the model using epochs
    for epoch in range(n_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch.view(-1, 1))
            loss.backward()
            optimizer.step()

        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_validation)
        mse = loss_fn(y_pred, y_validation)
        mse = float(mse)
        print(f'Epoch {epoch}. MSE: {mse:.2f}')
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())

    # Restore model and return best accuracy
    model.load_state_dict(best_weights)
    print(f"Best MSE: {best_mse:.2f}")
    print(f"Best RMSE: {np.sqrt(best_mse):.2f}")
    # Plot the history training
    plt.plot(history)
    model_history_filepath = r'./data/train/neural_network_pytorch_model_history_training.jpg'
    plt.savefig(model_history_filepath)

    # Copy Input dataset to generate report
    predictions_df = X.copy()
    predictions_df['price'] = y

    # Calculate scores
    model.eval()
    with torch.no_grad():  # Disabling gradient calculation
        y_validation_predictions = model(X_validation)
        r2 = r2_score(y_validation, y_validation_predictions)
        mse = mean_squared_error(y_validation, y_validation_predictions)
        mae = mean_absolute_error(y_validation, y_validation_predictions)
        results_json = {
            'r2': r2,
            'mse': mse,
            'mae': mae,
        }
        print("results:", results_json)
        # Build the training report results
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_X_predictions = model(X_tensor)
        predictions_df['prediction_price'] = y_X_predictions

    # Save model
    model_filepath = r'./data/train/neural_network_pytorch_model_cars_price_prediction.pth'
    torch.save(model.state_dict(), model_filepath)

    # Save results
    train_filepath = r'./data/train/neural_network_pytorch_cars_price_validation_prediction.csv'
    predictions_df.to_csv(train_filepath, index=False)

    print("Training Forward Neural Network - PyTorch Completed")

    # Return trained model and model results
    return model_filepath, results_json
