"""
This is a boilerplate pipeline 'model_serving'
generated using Kedro 0.19.12
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import requests 
from typing import Dict, List
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix,roc_curve, roc_auc_score
from sklearn.metrics import log_loss, roc_auc_score, precision_score, accuracy_score, recall_score,f1_score
import mlflow

def serve_and_predict(data: pd.DataFrame,model_name, model, port: int = 5002) -> pd.DataFrame:
    """
    faz a requisição pra API e retorna as previsões como DataFrame.
    """
    # Remover a shot_made_flag (se estiver presente)
    if 'shot_made_flag' in data.columns:
        data = data.drop(columns=['shot_made_flag'])
        print("Coluna 'shot_made_flag' removida do dataset de produção.")

    # Pegar os nomes das colunas
    model_columns = list(data.columns)
    print(f"Colunas enviadas pro modelo: {model_columns}")
    print(f"Número de colunas: {len(model_columns)}")

    # Converter os dados pra lista de listas
    data_values = data.values.tolist()
    print(f"Primeira linha de dados: {data_values[0]}")
    print(f"Número de colunas nos dados: {len(data_values[0])}")

    # Criar o JSON no formato que o MLflow espera
    request_data = {
        "dataframe_split": {
            "columns": model_columns,
            "data": data_values
        }
    } 
  
    # Enviar a requisição pro endpoint do MLflow
    print(f"Enviando requisição pra http://localhost:{port}/invocations...")
    response = requests.post(
        f"http://localhost:{port}/invocations",
        headers={"Content-Type": "application/json"},
        json=request_data
    )
 

    # Verificar a resposta
    if response.status_code == 200:
        print("Previsões recebidas com sucesso:")
        predictions = response.json()['predictions']
        print("Distribuição das previsões:")
        print(pd.Series(predictions).value_counts(normalize=True))
        
        # Retornar as previsões como DataFrame
        return pd.DataFrame({'shot_made_flag_pred': predictions})
    else:
        print(f"Erro na requisição: {response.status_code}")
        print(response.text)
        raise RuntimeError("Falha ao consumir a API do MLflow.")
 


def plot_shot_predictions_and_metrics(data: pd.DataFrame, predictions: pd.DataFrame, output_path: str, model, model_name: str):
    """
    Gera gráficos de dispersão, matriz de confusão, curva ROC e tabela de métricas
    para avaliação do modelo.
    """
 # Verificar se lat e lon estão no dataset
    if 'lat' not in data.columns or 'lon' not in data.columns:
        raise ValueError("As colunas 'lat' e 'lon' devem estar presentes no dataset para gerar o gráfico.")

    # Combinar os dados com as previsões
    plot_data = data[['lat', 'lon']].copy()
    plot_data['prediction'] = predictions['shot_made_flag_pred']

    # Separar os dados em acertos (1) e erros (0)
    acertos = plot_data[plot_data['prediction'] == 1]
    erros = plot_data[plot_data['prediction'] == 0]

    print(f"Qtde Acertos {model_name} === > ")
    print(acertos.count())

    print(f"Qtde erros {model_name}=== > ")
    print(erros.count())

    # Criar o gráfico de dispersão
    plt.figure(figsize=(10, 8))

    # Plotar os erros (0) com bolinhas vermelhas
    if not erros.empty:
        plt.scatter(
            erros['lon'],
            erros['lat'],
            c='red',
            marker='o',  # Bolinha
            s=50,  # Tamanho da bolinha
            label='Errou (0)',
            alpha=0.6
        )

    # Plotar os acertos (1) com bolinhas verdes
    if not acertos.empty:
        plt.scatter(
            acertos['lon'],
            acertos['lat'],
            c='green',
            marker='o',  # Bolinha
            s=50,  # Tamanho da bolinha
            label='Acertou (1)',
            alpha=0.6
        )

    # Adicionar título e labels
    plt.title("Local dos Arremessos do Kobe com Previsões do Modelo", fontsize=14)
    plt.xlabel("Longitude (lon)", fontsize=12)
    plt.ylabel("Latitude (lat)", fontsize=12)
    plt.legend(title="Previsão")

    # Salvar o gráfico
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo em: {output_path}")
    plt.close()   


    # Verificar se as colunas 'lat' e 'lon' estão presentes
    if 'lat' not in data.columns or 'lon' not in data.columns:
        raise ValueError("As colunas 'lat' e 'lon' devem estar presentes no dataset para gerar o gráfico.")

    # Combinar os dados com as previsões
    plot_data = data[['lat', 'lon']].copy()
    plot_data['prediction'] = predictions['shot_made_flag_pred']

    # Separar acertos (1) e erros (0)
    acertos = plot_data[plot_data['prediction'] == 1]
    erros = plot_data[plot_data['prediction'] == 0]

    # Criar o gráfico de dispersão
    plt.figure(figsize=(10, 8))
    if not erros.empty:
        plt.scatter(erros['lon'], erros['lat'], c='red', marker='o', s=50, label='Errou (0)', alpha=0.6)
    if not acertos.empty:
        plt.scatter(acertos['lon'], acertos['lat'], c='green', marker='o', s=50, label='Acertou (1)', alpha=0.6)
    
    plt.title("Local dos Arremessos com Previsões do Modelo")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="Previsão")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo em: {output_path}")
    plt.close()
    
    # Carregar o modelo do MLflow Model Registry 

    # Separar variáveis preditoras e alvo
    X_test = data.drop(columns=['shot_made_flag'])
    y_test = data['shot_made_flag']
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "log_loss": log_loss(y_test, y_proba)
    }
    metrics_df = pd.DataFrame([metrics])
    y_proba_df = pd.DataFrame(y_proba, columns=["probability"])
    
    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred: 0", "Pred: 1"], yticklabels=["True: 0", "True: 1"])
    plt.title("Matriz de Confusão")
    plt.ylabel("Classe Verdadeira")
    plt.xlabel("Classe Predita")
    plt.savefig(f"data/08_reporting/confusion_matrix_report_{model_name}_prod.png")
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.savefig(f"data/08_reporting/roc_curve_report_{model_name}_prod.png")
    
    # Gráfico de métricas
    plt.figure(figsize=(6, 3))
    sns.heatmap(metrics_df, annot=True, cmap="Blues", cbar=False, linewidths=1, linecolor='gray')
    plt.title("Relatório de Métricas do Modelo")
    plt.savefig(f"data/08_reporting/metrics_report_table_{model_name}_prod.png", bbox_inches="tight")
    
    return {
        "metrics": metrics_df,
        "predicted_probabilities": y_proba_df
    }
    