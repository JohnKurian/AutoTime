from flask import Flask
from flask_cors import CORS
import mlflow

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from flask import request

mlflow.set_tracking_uri("http://localhost:5000")


client = MlflowClient()





app = Flask(__name__)
CORS(app)

@app.route('/hello')
def say_hello_world():
    return {'result': "Hello World"}



@app.route('/create_experiment')
def create_experiment():
    experiment = mlflow.create_experiment('test_experiment_2')
    return {'result': experiment}



@app.route('/get_experiment')
def get_experiment():
    return {'result': "Hello experiment"}



@app.route('/get_runs')
def get_runs():
    experiment_id = request.args.get('experiment_id')

    runs = [{'run_id': run.run_id} for run in client.list_run_infos(experiment_id)]

    return {'result': runs}



@app.route('/runs')
def get_run():
    run_id = request.args.get('run_id')
    run_obj = client.get_run(run_id)
    run = [{'metrics': run_obj.data.metrics,
           'params': run_obj.data.params,
           'tags': run_obj.data.tags
           }]

    print(run)
    return {'result': run}








@app.route('/experiments')
def experiments():
    exps = [{'experiment_id': e.experiment_id,
             'name': e.name}
            for e in client.list_experiments(view_type=ViewType.ALL)]

    return {'result': exps}