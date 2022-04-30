## Context

Now that we have built a simple model, we want to make it better! The ultimate goal is to have a model that makes more accurate predictions on the test set, hence getting a RMSE as low as possible.

**So what can we do?**

We can try to improve our model by:
- building and trying to use different or additional features
- testing different estimators (linear, non linear, etc)
- tuning the hyperparameters of our pipeline

The problem is that it is often hard to keep track of all these experimentations. There are many different parameters that we can tune and many different combinations.

👉 **[MLflow](https://www.mlflow.org/docs/latest/concepts.html) is a very useful tool to help us in machine learning models iteration.**

In this series of exercise, you will get hands on using the [MLflow Tracking API](https://www.mlflow.org/docs/latest/tracking.html) in order to experiment with different features, models and parameters.

We now have a good workflow for bringing improvements to our model. We want to track all of our experiments and to be able to save the training runs and compare their performance. This is what MLflow Tracking is about.

## Setup

Copy `ml_flow_test.py` from the challenge directory to the directory of your packaged project.

<details>
  <summary markdown='span'><strong>💡 How to copy `ml_flow_test.py` from the challenge to the packaged project ? </strong></summary>


```bash
cp -r ~/code/<user.github_nickname>/<program.challenges_repo_name>/07-Data-Engineering/02-ML-Iteration/04-MLFlow-quickstart/ml_flow_test.py ~/code/<user.github_nickname>/TaxiFareModel
```

</details>

## MLflow quickstart

Install MLflow:

``` bash
pip install mlflow
```

Now open a new terminal window in your packaged project directory and run the content of `ml_flow_test.py`:

``` bash
cd ~/code/<user.github_nickname>/TaxiFareModel
python ml_flow_test.py
```

Explore the content of `ml_flow_test.py`. By running this file you just created 1 experiment having 2 runs each containing 1 metric and 1 param.

In order to explore the results, launch the MLflow UI in the same directory:

``` bash
mlflow ui
```

👉 And have a look in the interface: http://localhost:5000/

You just saved your first experiment in a local MLflow server on your machine 🎉

That is just fine, but what if you want to share your results with your team or want to train your model on another machine... How will you be able to contact your machine then?

## Hosted MLflow server

In order to solve this issue, we will be using a hosted MLflow server. The server will be reachable from any machine on the Internet. It will be able to store the results of experiments resulting from the training of models on any machine connected to the Internet. Anyone on the Internet will be able to observe the results of the experiments.

We will be using a Le Wagon hosted MLflow server in order to do this : https://mlflow.lewagon.ai/

Let's log the same parameters on the remote MLflow server. In order to do that, we will slightly modify `ml_flow_test.py`:

``` python
import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "test_experiment"

# Indicate mlflow to log to remote server
mlflow.set_tracking_uri("https://mlflow.lewagon.ai/")

client = MlflowClient()

try:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
except BaseException:
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

yourname = None

if yourname is None:
    print("please define your name, il will be used as a parameter to log")

for model in ["linear", "Randomforest"]:
    run = client.create_run(experiment_id)
    client.log_metric(run.info.run_id, "rmse", 4.5)
    client.log_param(run.info.run_id, "model", model)
    client.log_param(run.info.run_id, "student_name", yourname)
```

Now, run the code once again:

``` bash
python ml_flow_test.py
```

Let's have a look at the results: https://mlflow.lewagon.ai/

What's that ? Are you having trouble finding out your results or the "test_experiment" amongst the myriads of other experiments created by all the students of the bootcamp ?

👉 This is why we highly encourage you to create a custom experiment name matching the `[country code] [city] [login] model name + version` naming convention

By the way, here is [the default experiment](https://mlflow.lewagon.ai/#/experiments/94) 🎉

## Integrating MLflow to our packaged project

No that we are up to speed with MLflow, let's see how to integrate it in a project.

We will add a few methods to our existing code in order to start logging training runs.

Let's add the following methods to our `Trainer` class (do not forget the required imports):

``` python
class Trainer():

    # ...

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
```

Notice how [@memoized_property](https://pypi.org/project/memoized-property/) is actually quite powerful:
- It is a decorator
- It defines its decorated function as a [property](https://www.geeksforgeeks.org/python-property-function/)
- The property is memoized <=> only set at first call

Once the code is appended to our trainer, we can log parameters and metrics in MLflow from any method in our `Trainer` class by using:
- `self.mlflow_log_param(param_name, param_value)`
- `self.mlflow_log_metric(metric_name, metric_value)`

One last thing: we need to define the URL of the MLflow server and the name of our experiment in our class :

``` python
MLFLOW_URI = "https://mlflow.lewagon.ai/"
EXPERIMENT_NAME = "[country code] [city] [login] model name + version"  # 🚨 replace with your country code, city, github_nickname and model name and version
```

Let's log a few things in our class:
- Log the name of the estimator used for the training
- Log the RMSE of the trained model

Once the code is up, just run it:

``` bash
python -m TaxiFareModel.trainer
```

... And verify that it logged correctly on https://mlflow.lewagon.ai/

<details>
  <summary markdown='span'><strong> 💡 Finding your experiment easily in https://mlflow.lewagon.ai/ </strong></summary>

The easiest way to find your experiment without scrolling through the list of experiments is by using its `id`.

In order to do that, you can retrieve the `id` of the experiment in your code after the model has been trained for example:

``` python
experiment_id = trainer.mlflow_experiment_id

print(f"experiment URL: https://mlflow.lewagon.ai/#/experiments/{experiment_id}")
```

Another option is to use the name of your experiment in order to retrieve its id.

🚨 Replace the content of `EXPERIMENT_NAME` before running this in `ipython` for example.

``` python
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("https://mlflow.lewagon.ai/")

EXPERIMENT_NAME = "test_experiment"
experiment_id = MlflowClient().get_experiment_by_name(EXPERIMENT_NAME).experiment_id
print(f"experiment URL: https://mlflow.lewagon.ai/#/experiments/{experiment_id}")
```

</details>
