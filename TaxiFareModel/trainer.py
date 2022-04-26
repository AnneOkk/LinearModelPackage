from encoders import DistanceTransformer, TimeFeaturesEncoder
from utils import compute_rmse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import data_getter
import pandas as pd

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

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
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])


    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X, self.y)
        return self.pipeline

    def evaluate(self, X, y):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X)
        rmse = compute_rmse(y_pred, y)
        return rmse


if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    df = data_getter.get_data()
    df_clean = data_getter.clean_data(df)
    y = df_clean["fare_amount"]
    X = df_clean.drop("fare_amount", axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    inst = Trainer(X = X_train, y = y_train)
    pipeline = inst.set_pipeline()
    inst.run()
    rmse = inst.evaluate(X = X_val, y = y_val)
    print(rmse)
