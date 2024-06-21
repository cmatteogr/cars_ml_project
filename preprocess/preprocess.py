# import project libraries
from sklearnex import patch_sklearn

patch_sklearn()

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction import FeatureHasher
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
import re

from constants import RELEVANT_PREPROCESS_COLUMNS


# Apply msrp value
def map_msrp(msrp):
    """
    Replace 0 values by null

    :param msrp: manufacturer's suggested retail price
    """
    if msrp == 0:
        return np.nan
    return msrp


def clean_exterior_color(exterior_color):
    # Check if value is empty
    if pd.isna(exterior_color):
        return 'unknown'
    # Convert interior_color to lower case
    exterior_color = exterior_color.lower()
    # Remove special characters
    exterior_color = re.sub(r'[\W_+w/\/]', ' ', exterior_color)
    # Remove double spaces
    exterior_color = re.sub(r'\s+', ' ', exterior_color)
    # Apply trim 
    exterior_color = exterior_color.strip()
    # Return formated text
    return exterior_color


def get_interior_color_phrase_vector(exterior_color_phrase, model):
    exterior_color_words = exterior_color_phrase.split()
    exterior_color_word_vectors = [model.wv[word] for word in exterior_color_words if word in model.wv]
    if not exterior_color_word_vectors:
        print(f"No words found in model for phrase: {exterior_color_phrase}")
        return np.nan
    return sum(exterior_color_word_vectors) / len(exterior_color_word_vectors)


def clean_interior_color(interior_color):
    # Check if value is empty
    if pd.isna(interior_color):
        return 'unknown'
    # Convert interior_color to lower case
    interior_color = interior_color.lower()
    # Remove special characters
    interior_color = re.sub(r'[\W_+w/\/]', ' ', interior_color)
    # Remove double spaces
    interior_color = re.sub(r'\s+', ' ', interior_color)
    # Return formated text
    return interior_color


def get_interior_color_phrase_vector(interior_color_phrase, model):
    interior_color_words = interior_color_phrase.split()
    interior_color_word_vectors = [model.wv[word] for word in interior_color_words if word in model.wv]
    if not interior_color_word_vectors:
        print(f"No words found in model for phrase: {interior_color_phrase}")
        return np.nan
    return sum(interior_color_word_vectors) / len(interior_color_word_vectors)


def map_drivetrain(drivetrain):
    """
    Group the drive trian by categories

    :param drivetrain: Car drive train

    :return: Grouped drive train
    """
    if pd.isna(drivetrain):
        return np.nan
    # Apply lower case and replace special characters
    drivetrain = str(drivetrain).lower().replace('-', ' ')

    match drivetrain:
        case 'all wheel drive' | 'four wheel drive' | 'awd' | '4wd' | '4x2' | 'all wheel drive with locking and limited slip differential' | '4matic':
            return 'All-wheel Drive'
        case 'rear wheel drive' | 'rwd':
            return 'Rear-wheel Drive'
        case 'front wheel drive' | 'fwd' | 'front wheel drive':
            return 'Front-wheel Drive'
        case 'unknown':
            return np.nan
        case _:
            raise Exception(f"No expected drive train: {drivetrain}")


def clean_cat(cat):
    # Check if value is empty
    if pd.isna(cat):
        return 'unknown'
    # Convert cat to lower case
    cat = cat.lower()
    # Split by '_' and join again by ' '
    cat = ' '.join(cat.split('_'))
    # Remove double spaces
    cat = re.sub(r'\s+', ' ', cat)
    # Return formated text
    return cat


# Calculate the vectors feature avegare
def get_cat_phrase_vector(cat_phrase, model):
    cat_words = cat_phrase.split()
    cat_word_vectors = [model.wv[word] for word in cat_words if word in model.wv]
    if not cat_word_vectors:
        print(f"No words found in model for phrase: {cat_phrase}")
        return np.nan
    return sum(cat_word_vectors) / len(cat_word_vectors)


def map_fuel_type(fuel_type):
    """
    Group by fuel types

    :param fuel_type: Car fuel type

    :return Fuel type category
    """
    if pd.isna(fuel_type):
        return np.nan

    match fuel_type:
        case 'Gasoline' | 'Gasoline Fuel' | 'Diesel' | 'Premium Unleaded' | 'Regular Unleaded' | 'Premium Unleaded' | 'Diesel Fuel':
            return 'Gasoline'
        case 'Electric' | 'Electric with Ga':
            return 'Electric'
        case 'Hybrid' | 'Plug-In Hybrid' | 'Plug-in Gas/Elec' | 'Gas/Electric Hyb' | 'Hybrid Fuel' | 'Bio Diesel' | 'Gasoline/Mild Electric Hybrid' | 'Natural Gas':
            return 'Hybrid'
        case 'Flexible Fuel' | 'E85 Flex Fuel' | 'Flexible':
            return 'Flexible'
        case _:
            print(f"No expected fuel type: {fuel_type}")
            return np.nan


def map_stock_type(stock_type):
    """
    Map stock_type

    :param stock_type: stock type New/Used

    :return Binary stock_type 
    """
    if pd.isna(stock_type):
        return np.nan

    match stock_type:
        case 'New':
            return True
        case 'Used':
            return False
        case _:
            raise Exception(f"No expected stock type: {stock_type}")


def preprocess(cars_filepath, test_size=0.2, price_threshold=1500, make_frecuency_threshold=300,
               model_hash_batch_size=20, exterior_color_vector_size=5, interior_color_vector_size=5,
               cat_vector_size=3, train_inputer=False, isolation_forest_contamination=0.1, scale_data=False):
    """
    Pre process cars data

    :param cars_filepath: Cars datasource filepath
    :param test_size: Test size to split dataset
    :param price_threshold: Price min value threshold
    :param make_frecuency_threshold: Make category min frecuency value
    :param model_hash_batch_size: Model hash batch size
    :param exterior_color_vector_size: exterior_color vector size
    :param interior_color_vector_size: interior_color vector size
    :param cat_vector_size: cat vector size
    :param train_inputer: Indicates whether imputer model is trained or not
    :param isolation_forest_contamination: Outlier removal contamination
    :param scale_data: Scale data using min max scaler

    :return: X_train, y_train, X_test, y_test, impter model, outlier removal model
    
    """
    print("Star preprocess")

    print("####### Collect data")
    # Read CSV file
    print("Read data from data source")
    cars_df = pd.read_csv(cars_filepath)
    # Assign index
    cars_df.index = cars_df['listing_id']

    print("####### Clean data")
    # Remove duplicates
    print("Remove duplicates")
    cars_df.drop_duplicates(subset='listing_id', inplace=True)
    print(f"Dataframe shape after remove duplicates: {cars_df.shape}")
    # Filter relevant features
    print("Remove irrelevant features")
    cars_df = cars_df[RELEVANT_PREPROCESS_COLUMNS]
    # Remove NaN target
    print("Remove rows with empty target")
    cars_df = cars_df.loc[~cars_df['price'].isna()]
    # Remove cars with price under threshold
    print(f"Remove rows under target threshold: {price_threshold}")
    cars_df = cars_df.loc[cars_df['price'] >= price_threshold]
    # Remove NaN drivetrain
    print("Remove rows with empty drivetrain")
    cars_df = cars_df.loc[~cars_df['drivetrain'].isna()]
    # Remove NaN fuel_type
    print("Remove rows with empty fuel_type")
    cars_df = cars_df[~cars_df['fuel_type'].isna()]

    # Remove make values with low count
    print("Remove make values with low count")
    # Define the threshold for category frequency
    # Compute the frequency of each category
    make_category_counts = cars_df['make'].value_counts()
    # Identify categories that exceed the threshold
    make_categories_to_remove = make_category_counts[make_category_counts > make_frecuency_threshold].index
    # Filter the DataFrame to exclude rows with these categories
    cars_df = cars_df[cars_df['make'].isin(make_categories_to_remove)]

    print(f"Data set shape: {cars_df.shape}")

    # Split Train - Test dataset
    y = cars_df.pop('price')  # Target variable - price
    X = cars_df  # Car Features 

    print(f"Split Dataset train-test. train={1 - test_size}, test={test_size}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    y_train.index = X_train.index
    y_test.index = X_test.index

    print("####### Transfom data")
    # ### Apply Features transformation
    # Apply msrp transformation
    print("Apply msrp transformation")
    X_train['msrp'] = X_train['msrp'].map(map_msrp)
    X_test['msrp'] = X_test['msrp'].map(map_msrp)

    # Apply model transformation
    print("Apply model transformation")
    train_model_data = X_train['model'].apply(lambda x: {x: 1}).tolist()
    test_model_data = X_test['model'].apply(lambda x: {x: 1}).tolist()
    # Define the number of hash space
    n_hash = int(len(X_train['model'].unique()) / model_hash_batch_size)  # This values is a hyperparameter
    # Initialize FeatureHasher
    hasher = FeatureHasher(n_features=n_hash, input_type='dict')
    # Apply FeatureHasher
    train_model_hashed = hasher.transform(train_model_data)
    test_model_hashed = hasher.transform(test_model_data)
    # Generate model hashed dataframe
    train_model_hashed_df = pd.DataFrame(train_model_hashed.toarray(),
                                         columns=[f'model_hashed_{i}' for i in range(train_model_hashed.shape[1])],
                                         index=X_train.index)
    test_model_hashed_df = pd.DataFrame(test_model_hashed.toarray(),
                                        columns=[f'model_hashed_{i}' for i in range(test_model_hashed.shape[1])],
                                        index=X_test.index)
    # Concatenate the dataframes
    X_train = pd.concat([X_train, train_model_hashed_df], axis=1)
    X_test = pd.concat([X_test, test_model_hashed_df], axis=1)

    # Drop the model feature
    X_train.drop(columns='model', inplace=True)
    X_test.drop(columns='model', inplace=True)

    # Apply exterior_color transformation
    print("Apply exterior_color transformation")
    # Apply lower case and remove special characters
    X_train['exterior_color'] = X_train['exterior_color'].apply(clean_exterior_color)
    X_test['exterior_color'] = X_test['exterior_color'].apply(clean_exterior_color)
    # Tokenize colors sentences
    tokenized_exterior_color = [simple_preprocess(sentence) for sentence in X_train['exterior_color'].tolist()]
    # Train the Word2Vec model
    w2v_exterior_color_model = Word2Vec(sentences=tokenized_exterior_color, vector_size=exterior_color_vector_size,
                                        window=5, min_count=1, workers=4)
    # Calculate the vector for each interior color
    train_exterior_color_vectors_s = X_train['exterior_color'].apply(
        lambda ic: get_interior_color_phrase_vector(ic, w2v_exterior_color_model))
    test_exterior_color_vectors_s = X_test['exterior_color'].apply(
        lambda ic: get_interior_color_phrase_vector(ic, w2v_exterior_color_model))
    # Replace the nan values with an array of (0,0,0)
    base_invalid_value = [0] * exterior_color_vector_size
    train_exterior_color_vectors_s = train_exterior_color_vectors_s.apply(
        lambda x: x if isinstance(x, np.ndarray) else base_invalid_value)
    test_exterior_color_vectors_s = test_exterior_color_vectors_s.apply(
        lambda x: x if isinstance(x, np.ndarray) else base_invalid_value)
    # Generate the interior color df using the transformed feature vectors
    train_exterior_color_df = pd.DataFrame(train_exterior_color_vectors_s.values.tolist(),
                                           columns=[f'exterior_color_x{i}' for i in
                                                    range(len(train_exterior_color_vectors_s.iloc[0]))],
                                           index=X_train.index)
    test_exterior_color_df = pd.DataFrame(test_exterior_color_vectors_s.values.tolist(),
                                          columns=[f'exterior_color_x{i}' for i in
                                                   range(len(test_exterior_color_vectors_s.iloc[0]))],
                                          index=X_test.index)
    # Concatenate the dataframes
    X_train = pd.concat([X_train, train_exterior_color_df], axis=1)
    X_test = pd.concat([X_test, test_exterior_color_df], axis=1)

    # Once used drop the exterior_color feature
    X_train.drop(columns='exterior_color', inplace=True)
    X_test.drop(columns='exterior_color', inplace=True)

    # Apply interior_color transformation
    print("Apply interior_color transformation")
    # Apply lower case and remove special characters
    X_train['interior_color'] = X_train['interior_color'].apply(clean_interior_color)
    X_test['interior_color'] = X_test['interior_color'].apply(clean_interior_color)
    # Tokenize colors sentences
    tokenized_interior_color = [simple_preprocess(sentence) for sentence in X_train['interior_color'].tolist()]
    # Train the Word2Vec model
    w2v_interior_color_model = Word2Vec(sentences=tokenized_interior_color, vector_size=interior_color_vector_size,
                                        window=5, min_count=1, workers=4)
    # Calculate the vertor for each interior color
    train_interior_color_vectors_s = X_train['interior_color'].apply(
        lambda ic: get_interior_color_phrase_vector(ic, w2v_interior_color_model))
    test_interior_color_vectors_s = X_test['interior_color'].apply(
        lambda ic: get_interior_color_phrase_vector(ic, w2v_interior_color_model))
    # Replace the nan values with an array of (0,0,0)
    base_invalid_value = [0] * interior_color_vector_size
    train_interior_color_vectors_s = train_interior_color_vectors_s.apply(
        lambda x: x if isinstance(x, np.ndarray) else base_invalid_value)
    test_interior_color_vectors_s = test_interior_color_vectors_s.apply(
        lambda x: x if isinstance(x, np.ndarray) else base_invalid_value)
    # Generate the interior color df using the transformed feature vectors
    train_interior_color_df = pd.DataFrame(train_interior_color_vectors_s.values.tolist(),
                                           columns=[f'interior_color_x{i}' for i in
                                                    range(len(train_interior_color_vectors_s.iloc[0]))],
                                           index=X_train.index)
    test_interior_color_df = pd.DataFrame(test_interior_color_vectors_s.values.tolist(),
                                          columns=[f'interior_color_x{i}' for i in
                                                   range(len(test_interior_color_vectors_s.iloc[0]))],
                                          index=X_test.index)
    # Concatenate the dataframes
    X_train = pd.concat([X_train, train_interior_color_df], axis=1)
    X_test = pd.concat([X_test, test_interior_color_df], axis=1)
    # Once used drop the interior_color feature
    X_train.drop(columns='interior_color', inplace=True)
    X_test.drop(columns='interior_color', inplace=True)

    # Applt drive train transformation
    print("Apply drivetrain transformation")
    X_train['drivetrain'] = X_train['drivetrain'].map(map_drivetrain)
    X_test['drivetrain'] = X_test['drivetrain'].map(map_drivetrain)
    # Initialize the OneHotEncoder drivetrain
    drivetrain_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # Fit and transform the data
    ohe_drivetrain_model = drivetrain_encoder.fit(X_train[['drivetrain']])
    train_drivetrain_encoded_data = ohe_drivetrain_model.transform(X_train[['drivetrain']])
    test_drivetrain_encoded_data = ohe_drivetrain_model.transform(X_test[['drivetrain']])
    # Convert the drivetrain encoded data into a DataFrame
    train_drivetrain_encoded_df = pd.DataFrame(train_drivetrain_encoded_data,
                                               columns=drivetrain_encoder.get_feature_names_out(['drivetrain']),
                                               index=X_train.index)
    test_drivetrain_encoded_df = pd.DataFrame(test_drivetrain_encoded_data,
                                              columns=drivetrain_encoder.get_feature_names_out(['drivetrain']),
                                              index=X_test.index)
    # Concatenate the original DataFrame with the drivetrain encoded DataFrame
    X_train = pd.concat([X_train, train_drivetrain_encoded_df], axis=1)
    X_test = pd.concat([X_test, test_drivetrain_encoded_df], axis=1)

    # Once used drop the interior_color feature
    X_train.drop(columns='drivetrain', inplace=True)
    X_test.drop(columns='drivetrain', inplace=True)

    # Apply make transformation
    print("Apply make transformation")
    # Initialize the OneHotEncoder maker
    make_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # Fit and transform the data
    ohe_make_model = make_encoder.fit(X_train[['make']])
    train_make_encoded_data = ohe_make_model.transform(X_train[['make']])
    test_make_encoded_data = ohe_make_model.transform(X_test[['make']])
    # Convert the drivetrain encoded data into a DataFrame
    train_make_encoded_df = pd.DataFrame(train_make_encoded_data, columns=make_encoder.get_feature_names_out(['make']),
                                         index=X_train.index)
    test_make_encoded_df = pd.DataFrame(test_make_encoded_data, columns=make_encoder.get_feature_names_out(['make']),
                                        index=X_test.index)
    # Concatenate the original DataFrame with the drivetrain encoded DataFrame
    X_train = pd.concat([X_train, train_make_encoded_df], axis=1)
    X_test = pd.concat([X_test, test_make_encoded_df], axis=1)

    # Once used drop the make feature
    X_train.drop(columns='make', inplace=True)
    X_test.drop(columns='make', inplace=True)

    # Apply bodystyle transformation
    print("Apply bodystyle transformation")
    # Initialize the OneHotEncoder bodystyle
    bodystyle_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # Fit and transform the data
    ohe_bodystyle_model = bodystyle_encoder.fit(X_train[['bodystyle']])
    train_bodystyle_encoded_data = ohe_bodystyle_model.fit_transform(X_train[['bodystyle']])
    test_bodystyle_encoded_data = ohe_bodystyle_model.fit_transform(X_test[['bodystyle']])
    # Convert the drivetrain encoded data into a DataFrame
    train_bodystyle_encoded_df = pd.DataFrame(train_bodystyle_encoded_data,
                                              columns=bodystyle_encoder.get_feature_names_out(['bodystyle']),
                                              index=X_train.index)
    test_bodystyle_encoded_df = pd.DataFrame(test_bodystyle_encoded_data,
                                             columns=bodystyle_encoder.get_feature_names_out(['bodystyle']),
                                             index=X_test.index)
    # Concatenate the original DataFrame with the drivetrain encoded DataFrame
    X_train = pd.concat([X_train, train_bodystyle_encoded_df], axis=1)
    X_test = pd.concat([X_test, test_bodystyle_encoded_df], axis=1)

    # Once used drop the interior_color feature
    X_train.drop(columns='bodystyle', inplace=True)
    X_test.drop(columns='bodystyle', inplace=True)

    # Apply cat transformation
    print("Apply cat transformation")
    # Apply lower case and remove special characters
    X_train['cat'] = X_train['cat'].apply(clean_cat)
    X_test['cat'] = X_test['cat'].apply(clean_cat)
    # Tokenize colors sentences
    tokenized_cat = [simple_preprocess(sentence) for sentence in X_train['cat'].tolist()]
    # Train the Word2Vec model
    w2v_cat_model = Word2Vec(sentences=tokenized_cat, vector_size=cat_vector_size, window=5, min_count=1, workers=4)
    # Calculate the vertor for each cat
    train_cat_vectors_s = X_train['cat'].apply(lambda ic: get_cat_phrase_vector(ic, w2v_cat_model))
    test_cat_vectors_s = X_test['cat'].apply(lambda ic: get_cat_phrase_vector(ic, w2v_cat_model))
    # Replace the nan values with an array of (0,0,0)
    base_invalid_value = [0] * cat_vector_size
    train_cat_vectors_s = train_cat_vectors_s.apply(lambda x: x if isinstance(x, np.ndarray) else base_invalid_value)
    test_cat_vectors_s = test_cat_vectors_s.apply(lambda x: x if isinstance(x, np.ndarray) else base_invalid_value)
    # Generate the interior color df using the transformed feature vectors
    train_cat_data = pd.DataFrame(train_cat_vectors_s.values.tolist(),
                                  columns=[f'cat_x{i}' for i in range(len(train_cat_vectors_s.iloc[0]))],
                                  index=X_train.index)
    test_cat_data = pd.DataFrame(test_cat_vectors_s.values.tolist(),
                                 columns=[f'cat_x{i}' for i in range(len(test_cat_vectors_s.iloc[0]))],
                                 index=X_test.index)

    # Concatenate the dataframes
    X_train = pd.concat([X_train, train_cat_data], axis=1)
    X_test = pd.concat([X_test, test_cat_data], axis=1)

    # Remove cat column
    X_train.drop(columns='cat', inplace=True)
    X_test.drop(columns='cat', inplace=True)

    # Apply fuel type transformation
    print("Apply fuel_type transformation")
    X_train['fuel_type'] = X_train['fuel_type'].map(map_fuel_type)
    X_test['fuel_type'] = X_test['fuel_type'].map(map_fuel_type)
    # Remove invalid fuel type
    X_train = X_train.loc[~X_train['fuel_type'].isna()]
    X_test = X_test.loc[~X_test['fuel_type'].isna()]

    # Initialize the OneHotEncoder drivetrain
    eho_fuel_type_model = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # Fit and transform the data
    ohe_fuel_type_model = eho_fuel_type_model.fit(X_train[['fuel_type']])
    train_fuel_type_encoded_data = ohe_fuel_type_model.transform(X_train[['fuel_type']])
    test_fuel_type_encoded_data = ohe_fuel_type_model.transform(X_test[['fuel_type']])
    # Convert the drivetrain encoded data into a DataFrame
    train_fuel_type_encoded_df = pd.DataFrame(train_fuel_type_encoded_data,
                                              columns=eho_fuel_type_model.get_feature_names_out(['fuel_type']),
                                              index=X_train.index)
    test_fuel_type_encoded_df = pd.DataFrame(test_fuel_type_encoded_data,
                                             columns=eho_fuel_type_model.get_feature_names_out(['fuel_type']),
                                             index=X_test.index)
    # Concatenate the original DataFrame with the drivetrain encoded DataFrame
    X_train = pd.concat([X_train, train_fuel_type_encoded_df], axis=1)
    X_test = pd.concat([X_test, test_fuel_type_encoded_df], axis=1)

    # Once used drop the interior_color feature
    X_train.drop(columns='fuel_type', inplace=True)
    X_test.drop(columns='fuel_type', inplace=True)

    # Apply binary transformation
    print("Apply stock_type transformation")
    X_train['stock_type'] = X_train['stock_type'].map(map_stock_type)
    X_test['stock_type'] = X_test['stock_type'].map(map_stock_type)

    print("####### Imputate data")
    # train/use Imputer
    print("Apply Iterative imputation")
    # NOTE: This condition is needed because the imputation model take a long trainning
    imputer_model_filepath = r'./artifacts/preprocess_regression_imputer_model.pkl'
    if train_inputer:
        # Train imputer
        imp = IterativeImputer(estimator=RandomForestRegressor(), verbose=1)
        # fit on the dataset 
        imp.fit(X_train)
        # Save imnputer model
        joblib.dump(imp, imputer_model_filepath)

    # Load your model
    imp: IterativeImputer = joblib.load(imputer_model_filepath)

    # Apply imputation
    train_df_trans = imp.transform(X_train)
    test_df_trans = imp.transform(X_test)
    # transform the dataset 
    X_train = pd.DataFrame(train_df_trans, columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(test_df_trans, columns=X_test.columns, index=X_test.index)

    print("####### Remove outliers data")
    # ### Outliers Removal
    print("Apply Outlier Removal")
    iso_forest = IsolationForest(n_estimators=200, contamination=isolation_forest_contamination, random_state=42,
                                 verbose=1)
    # Fit the model
    iso_forest.fit(X_train)
    # Remove outliers 
    X_train['outlier'] = iso_forest.predict(X_train)
    X_test['outlier'] = iso_forest.predict(X_test)
    # Remove global outliers
    X_train = X_train[X_train['outlier'] != -1]
    X_test = X_test[X_test['outlier'] != -1]
    # Remove the outlier column
    X_train.drop(columns='outlier', inplace=True)
    X_test.drop(columns='outlier', inplace=True)
    # Filter targets
    y_train = y_train[y_train.index.isin(X_train.index)]
    y_test = y_test[y_test.index.isin(X_test.index)]

    # Scale data if needed
    scaler = None
    if scale_data:
        print("####### Scale data")
        # Init scaler model
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        print("Apply Scale Min/Max Transformation")
        # Apply scale transformation
        train_df_trans = scaler.transform(X_train)
        test_df_trans = scaler.transform(X_test)
        # transform the dataset
        X_train = pd.DataFrame(train_df_trans, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(test_df_trans, columns=X_test.columns, index=X_test.index)

    print(f"Cars train dataset size after preprocess: {X_train.shape}")
    print(f"Cars test dataset size after preprocess: {X_test.shape}")

    # Export preprocess data
    train_data_df = X_train.copy()
    train_data_df['price'] = y_train
    train_data_filepath = r'./data/preprocess/train_data.csv'
    train_data_df.to_csv(train_data_filepath)
    test_data_df = X_test.copy()
    test_data_df['price'] = y_test
    test_data_filepath = r'./data/preprocess/test_data.csv'
    test_data_df.to_csv(test_data_filepath)

    # Save preprocess models
    # Save drive train encoding model
    print("Save preprocess models")
    ohe_drivetrain_model_filepath = r'./artifacts/preprocess_ohe_drivetrain_model.pkl'
    joblib.dump(ohe_drivetrain_model, ohe_drivetrain_model_filepath)
    # Save make encoding model
    ohe_make_model_filepath = r'./artifacts/preprocess_ohe_make_model.pkl'
    joblib.dump(ohe_make_model, ohe_make_model_filepath)
    # Save bodystyle encoding model
    ohe_bodystyle_model_filepath = r'./artifacts/preprocess_ohe_bodystyle_model.pkl'
    joblib.dump(ohe_bodystyle_model, ohe_bodystyle_model_filepath)
    # Save bodystyle encoding model
    ohe_fuel_type_model_filepath = r'./artifacts/preprocess_ohe_fuel_type_model.pkl'
    joblib.dump(ohe_fuel_type_model, ohe_fuel_type_model_filepath)
    # Save exterior color encoding model
    w2v_exterior_color_model_filepath = r'./artifacts/preprocess_w2v_exterior_color_model.model'
    w2v_exterior_color_model.save(w2v_exterior_color_model_filepath)
    # Save interior color encoding model
    w2v_interior_color_model_filepath = r'./artifacts/preprocess_w2v_interior_color_model.model'
    w2v_interior_color_model.save(w2v_interior_color_model_filepath)
    # Save exterior color encoding model
    w2v_cat_model_filepath = r'./artifacts/preprocess_w2v_cat_model.model'
    w2v_cat_model.save(w2v_cat_model_filepath)
    # NOTE: imputer model was already saved
    # Save isolation forest model
    iso_forest_model_filepath = r'./artifacts/preprocess_outlier_detection_model.pkl'
    joblib.dump(iso_forest, iso_forest_model_filepath)
    # Save scaler model
    scaler_model_filepath = r'./artifacts/preprocess_scaler_model.pkl'
    joblib.dump(scaler, scaler_model_filepath)

    print("Preprocess completed")

    # Return model
    return (X_train, y_train, X_test, y_test, imputer_model_filepath, iso_forest_model_filepath, scaler_model_filepath,
            ohe_drivetrain_model_filepath, ohe_make_model_filepath, ohe_bodystyle_model_filepath,
            ohe_fuel_type_model_filepath, w2v_exterior_color_model_filepath, w2v_interior_color_model_filepath,
            w2v_cat_model_filepath)
