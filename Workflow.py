import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from typing import Any, Dict, List
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from pickle import dump
from prefect import task, flow

# Loading the Dataset
@task
def load_data(path:str, unwanted_col: List) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.drop(unwanted_col, axis = 1, inplace= True)
    return data

# Target classes
@task
def get_classes(target:pd.Series) -> List[str]:
    return list(target.unique())

# Splitting the Dataset into Train and Test
@task
def split_data(input:pd.DataFrame, output:pd.Series, test_data_ratio:float, random_state:int) -> Dict[str, Any]:
    X_tr, X_te, Y_tr, Y_te = train_test_split(input,output, test_size = test_data_ratio, random_state = random_state)
    return {'X_TRAIN':X_tr, 'X_TEST':X_te, 'Y_TRAIN':Y_tr, 'Y_TEST':Y_te}

# Scaler Func
@task
def get_scaler(x_train_num: pd.DataFrame) -> Any:
    scaler = StandardScaler()
    scaler.fit(x_train_num)
    return scaler

# Encoder Func
@task
def get_encoder(x_train_cat: pd.DataFrame, cat = List[list]) -> Any:
    encoder = OrdinalEncoder(categories = cat, dtype= np.int64)
    encoder.fit(x_train_cat)
    return encoder

# Scaling the numeric columns
@task
def rescale_data(scaler:Any, num_data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(scaler.transform(num_data), index= num_data.index, columns= num_data.columns)

# Encoding the classification columns
@task
def encode_data(encoder:Any, cat_data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(encoder.transform(cat_data), index= cat_data.index, columns= cat_data.columns)

# find the best model
@task
def find_best_model(X_train: pd.DataFrame, y_train: pd.Series, estimator: Any, parameter: List) -> Any:
    with mlflow.start_run() as run:
        mlflow.set_tag('Dev','Vishal')
        mlflow.set_tag('Algo', 'RandomForestReg')
        mlflow.log_param('data-path', 'data/diamonds.csv')

        RF_regressor = RandomForestRegressor()
        RF_regressor.fit(X_train, y_train)

        mlflow.sklearn.log_model(RF_regressor, artifact_path='models')

        run_id = run.info.run_id

    return RF_regressor, run_id

# Estimation of Model
@task
def estimate(run_id: str, RF_regressor: Any, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:

    with mlflow.start_run(run_id = run_id):
        y_test_pred = RF_regressor.predict(X_test)

        mse = metrics.mean_squared_error(y_test, y_test_pred)
        mae = metrics.mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))

        mlflow.log_metric('MSE', mse)
        mlflow.log_metric('MAE', mae)
        mlflow.log_metric('RMSE', rmse)

    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse}

def save_model(file: Any,string:str):
    dump(file, open(f'./models/{string}', 'wb'))

# Workflow
@flow
def main(path: str):

    # Database mlflow
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('Diamond Price Prediction')

    # Define Parameters
    TARGET_COL = 'price'
    UNWANTED_COL = []
    DATA_PATH = path
    TEST_DATA_RATIO = 0.20
    RANDOM_STATE = 42
    CATEGORIES = [['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'],
                  ['J','I','H','G','F','E','D'],
                  ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']]
    ESTIMATOR = RandomForestRegressor()
    PARAMETER = {'bootstrap': [True],
                'max_depth': [80, 90, 100, 110],
                'max_features': [2, 3],
                'min_samples_leaf': [3, 4, 5],
                'min_samples_split': [8, 10, 12],
                'n_estimators': [100, 200, 300, 1000]
                }

    # Load the Data
    dataframe = load_data(path = DATA_PATH, unwanted_col = UNWANTED_COL)

    # X, y split
    target_data = dataframe[TARGET_COL]
    input_data = dataframe.drop([TARGET_COL],axis = 1)

    # Unique target classes
    classes = get_classes(target = target_data)

    # split train and test dataset
    train_test_data = split_data(input= input_data, output= target_data, test_data_ratio= TEST_DATA_RATIO, random_state= RANDOM_STATE)

    # numeric categorical split
    x_train_num = train_test_data['X_TRAIN'].select_dtypes(['float64']) 
    x_train_cat = train_test_data['X_TRAIN'].select_dtypes(['object'])
    x_test_num = train_test_data['X_TEST'].select_dtypes(['float64'])
    x_test_cat = train_test_data['X_TEST'].select_dtypes(['object'])

    # get scaler and encoder
    scaler = get_scaler(x_train_num = x_train_num)
    encoder = get_encoder(x_train_cat = x_train_cat, cat = CATEGORIES)

    # rescale and encode the train data
    scaled_tr_data = rescale_data(scaler = scaler, num_data = x_train_num)
    encoded_tr_data = encode_data(encoder = encoder, cat_data = x_train_cat)

    # concat the scaled and encoded train data
    train_test_data['X_TRAIN'] = pd.concat([scaled_tr_data,encoded_tr_data], axis = 1)

    # rescale and encode the test data
    scaled_te_data = rescale_data(scaler = scaler, num_data = x_test_num)
    encoded_te_data = encode_data(encoder = encoder, cat_data = x_test_cat)

    # concat the scaled and encoded test data
    train_test_data['X_TEST'] = pd.concat([scaled_te_data,encoded_te_data], axis = 1)

    # Model Training
    rf_regressor, run_id = find_best_model(X_train= train_test_data['X_TRAIN'], y_train= train_test_data['Y_TRAIN'], estimator= ESTIMATOR, parameter= PARAMETER)
    
    # Estimate the model built
    estimator = estimate(run_id, rf_regressor, X_test = train_test_data['X_TEST'], y_test = train_test_data['Y_TEST'])

    print('Mean Squared Error :', estimator['MSE'])
    print('Mean Absolute Error :', estimator['MAE'])
    print('Root Mean Squared Error :', estimator['RMSE'])

    # saving the models
    save_model(scaler, 'StandardScaler.pkl')
    save_model(encoder, 'OrdinalEncoder.pkl')
    save_model(rf_regressor, 'RandomForest.pkl')


# Run the main function
main(path = './data/diamonds.csv')

# Deploy the main function using schedules in prefect
from prefect.deployments import Deployment
from datetime import timedelta
from prefect.server.schemas.schedules import IntervalSchedule

deployment = Deployment.build_from_flow(
    flow = main,
    name = "model_training",
    schedule=(IntervalSchedule(interval = timedelta(minutes=120))),
    work_queue_name = 'ml'
)

deployment.apply()


