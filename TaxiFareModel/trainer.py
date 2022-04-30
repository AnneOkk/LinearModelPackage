import joblib
from termcolor import colored
import mlflow
from TaxiFareModel.data_getter import get_data, clean_data, df_optimized
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from TaxiFareModel.params import STORAGE_LOCATION, BUCKET_NAME
from google.cloud import storage
from TaxiFareModel.gcp import storage_upload
from tpot import TPOTRegressor

MLFLOW_URI = "https://mlflow.lewagon.ai/"
EXPERIMENT_NAME = "[DE][BERLIN][AnneOkk]"


class Trainer(object):
    def __init__(self, X, y, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        # for MLFlow
        self.experiment_name = EXPERIMENT_NAME
        self.params = dict(nrows=10000,
                           upload=True,
                           local=False,  # set to False to get data from GCP (Storage or BigQuery)
                           gridsearch=False,
                           optimize=True,
                           estimator="xgboost",
                           mlflow=True,  # set to True to log params to mlflow
                           experiment_name=EXPERIMENT_NAME,
                           pipeline_memory=None, # None if no caching and True if caching expected
                           distance_type="manhattan",
                           feateng=["distance_to_center", "direction", "distance", "time_features", "geohash"],
                           n_jobs=-1) # Try with njobs=1 and njobs = -1

    def set_experiment_name(self, experiment_name):
        '''defines the experiment name for MLFlow'''
        self.experiment_name = experiment_name

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, [
                "pickup_latitude",
                "pickup_longitude",
                'dropoff_latitude',
                'dropoff_longitude'
            ]),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('tpot', TPOTRegressor(config_dict = 'TPOT sparse'))
        ])

    def run(self):
        self.set_pipeline()
        self.mlflow_log_param("model", "Linear")
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        print(self.pipeline.score(X_test, y_test))
        self.pipeline.export('tpot_exported_pipeline.py')
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric("rmse", rmse)
        return round(rmse, 2)


    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))
        print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")


# MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # Get and clean data
    N = 10000
    df = get_data(nrows=N)
    df = clean_data(df)
    df = df_optimized(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Train and save model, locally and
    trainer = Trainer(X=X_train, y=y_train, n_jobs=-1, nrows=10000)
    trainer.set_experiment_name(EXPERIMENT_NAME)
    trainer.run()
    rmse = trainer.evaluate(X_test, y_test)
    print(f"rmse: {rmse}")
    trainer.save_model()
    storage_upload(STORAGE_LOCATION, bucket=BUCKET_NAME, rm=False)
