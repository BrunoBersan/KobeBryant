import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
import mlflow
from datetime import date, datetime
import mlflow

def start_mlflow_run() -> dict:
    """Inicia o run pai do MLFlow com o nome do projeto."""
    experiment_name = "Projeto_Kobe_Kedro"
    run_name = "projeto_kobe"  # Nome fixo para o run "pai"

    # Define o experimento
    mlflow.set_experiment(experiment_name)

    # Inicia o run "pai"
    mlflow.start_run(run_name=run_name)
    mlflow.set_tag("stage", "pipeline_execution")
    print(f"Run pai iniciado: {run_name}")

    # Retorna um output fictício para satisfazer o Kedro
    return {"status": f"Run {run_name} iniciado"}

def end_mlflow_run(mlflow_run_status: dict) -> None:
    """Finaliza o run pai do MLFlow."""
    mlflow.end_run()
    print("Run pai finalizado.")


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame: 
    """Tratamento de valores nulos"""

    run_name = f"tratamento_dados_nulos_{date.today().strftime('%Y-%m-%d')}"
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.set_tag("project_name", "projeto_kobe")
        mlflow.set_tag("stage", "data_preparation_remove_null")
        return data.dropna()    


def remove_duplicates_and_validate(data: pd.DataFrame) -> pd.DataFrame: 
    """Remoção de duplicatas e validações diversas"""
      
    run_name = f"tratamento_dados_duplicados_{date.today().strftime('%Y-%m-%d')}"
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.set_tag("project_name", "projeto_kobe")
        mlflow.set_tag("stage", "preparacao_dados_duplicates")
        return data.drop_duplicates(keep='last')   
