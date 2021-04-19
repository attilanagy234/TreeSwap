import mlflow
import git


def log_metrics(self, run_name):
    try:
        repository = git.Repo(search_parent_directories=True)
        SHA = repository.head.object.hexsha
    except:
        SHA = "UNKNOWN"

    mlflow.start_run(run_name=run_name)
    mlflow.log_param('Commit hash', SHA)

    mlflow.start_run(run_name=run_name)
    mlflow.log_param('Commit hash', SHA)
    mlflow.log_param('How much do you like NLP?', 100)


if __name__ == '__main__':
    # Setup Mlflow
    run_name = "description_classifier_sum_rule_take_top1"  # Set manually each time you run the script
    experiment_name = "industry_mapping_evaluation"
    tracking_uri = 'sqlite:////mlruns.db'  # We should keep the mlruns.db in github
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    # mlflow.get_experiment(experiment_name)

    log_metrics('Example run name')
