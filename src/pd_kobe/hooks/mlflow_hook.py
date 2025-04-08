from kedro.framework.hooks import hook_impl
import mlflow

class MLflowHook:
    @hook_impl
    def before_pipeline_run(self, run_params):
        """Inicia o run pai do MLflow antes do pipeline."""
        experiment_name = "Projeto_Kobe_Kedro"
        run_name = "projeto_kobe" 

        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        mlflow.set_tag("stage", "pipeline_execution")

        print(f"Run pai iniciado: {run_name}")

    @hook_impl
    def after_pipeline_run(self, run_params):
        """Finaliza o run pai do MLflow após o pipeline."""
        mlflow.end_run()
        print("Run pai finalizado.")