import mlflow

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

mlflow.set_tracking_uri("http://localhost:5000")

def print_experiment_info(experiments):
    for e in experiments:
        print("- experiment_id: {}, name: {}, lifecycle_stage: {}"
              .format(e.experiment_id, e.name, e.lifecycle_stage))

client = MlflowClient()
# for name in ["Experiment 1", "Experiment 2"]:
#     exp_id = client.create_experiment(name)

# # Delete the last experiment
# client.delete_experiment(exp_id)

# Fetch experiments by view type
print("Active experiments:")
print_experiment_info(client.list_experiments(view_type=ViewType.ACTIVE_ONLY))
print("Deleted experiments:")
print_experiment_info(client.list_experiments(view_type=ViewType.DELETED_ONLY))
print("All experiments:")
print_experiment_info(client.list_experiments(view_type=ViewType.ALL))
print('here')

import pdb


# get runs for a particular experiment id

for run in client.list_run_infos('1'):
    print(run.start_time, run.end_time)
    print(run.run_id)
    print(run.artifact_uri)


    # get metrics params and tags for a run
    print('here')

    run_obj = client.get_run(run.run_id)

    print(client.get_run(run.run_id))
    print(client.list_artifacts(run.run_id))


#https://www.mlflow.org/docs/latest/search-syntax.html

from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType

run = MlflowClient().search_runs(
  experiment_ids="0",
  filter_string="",
  run_view_type=ViewType.ACTIVE_ONLY,
  max_results=1,
  order_by=["metrics.foo DESC"]
)[0]

print(run)



from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType

query = "params.model = 'CNN' and params.layers = '10' and metrics.`prediction accuracy` >= 0.945"
runs = MlflowClient().search_runs(
    experiment_ids=["3", "4"],
    filter_string=query,
    run_view_type=ViewType.ACTIVE_ONLY
)





from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType

all_experiments = [exp.experiment_id for exp in MlflowClient().list_experiments()]
runs = MlflowClient().search_runs(
    experiment_ids=all_experiments,
    filter_string="params.model = 'Inception'",
    run_view_type=ViewType.ALL
)



