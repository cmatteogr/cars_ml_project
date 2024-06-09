from pycaret.classification import load_model, predict_model
import pandas as pd


def inference(X):
    # Load the model
    model = load_model('../data/train/automl_model_cars_price_prediction')
    # Make predictions
    predictions = predict_model(model, data=X)

    # Display the predictions
    predictions.to_csv('../data/inference/predictions_car_price_prediction.csv', index=False)


test_df = pd.read_csv('../data/data_exploration/output/train_data.csv', index_col=0)
test_df.drop(columns='price', inplace=True)
inference(test_df)
