"""
Nós criados para plotar relatórios de avaliação dos modelos
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,roc_curve, roc_auc_score
import matplotlib.image as mpimg
import numpy as np
import requests
import subprocess
import time
import psutil
import os
from typing import Dict, List

def save_model_plots_metrics(metrics: pd.DataFrame,shots_data: pd.DataFrame, predicted_probabilities: pd.DataFrame, predictions: pd.DataFrame, model, data_type: str):

    model_name = get_model_short_name(model)
    print(f"Modelo recebido: {type(model).__name__} (short name: {model_name})_{data_type}")
    print(f"ID do objeto modelo: {id(model)}_{data_type}")
    print('CHEGOU AQUI ==> ' + model_name)

    # Criar o gráfico
    cm = confusion_matrix(predictions["actual"], predictions["predicted"])

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred: 0", "Pred: 1"], yticklabels=["True: 0", "True: 1"])
    plt.title("Matriz de Confusão")
    plt.ylabel("Classe Verdadeira")
    plt.xlabel("Classe Predita")
    plt.savefig(f"data/08_reporting/confusion_matrix_report_{model_name}_{data_type}.png")

    # Calcular a Curva ROC
    fpr, tpr, _ = roc_curve(predictions["actual"], predicted_probabilities["probability"])
    roc_auc = roc_auc_score(predictions["actual"], predicted_probabilities["probability"])

    # Plotar a Curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.savefig(f"data/08_reporting/roc_curve_report_{model_name}_{data_type}.png")

    # Criar um gráfico com a tabela
    plt.figure(figsize=(6, 3))
    sns.heatmap(metrics, annot=True, cmap="Blues", cbar=False, linewidths=1, linecolor='gray')
    # Título do gráfico
    plt.title("Relatório de Métricas do Modelo", fontsize=16)
    # Mostrar o gráfico
    plt.savefig(f"data/08_reporting/metrics_report_table_{model_name}_{data_type}.png", bbox_inches="tight")

    # Visualizar as distribuições de probabilidades
    plt.figure(figsize=(8, 6))
    sns.histplot(predicted_probabilities['probability'], kde=True, bins=30, color="blue")
    plt.title("Distribuição das Probabilidades Preditas")
    plt.xlabel("Probabilidade")
    plt.ylabel("Frequência")
    plt.savefig(f"data/08_reporting/distribuitions_{model_name}_{data_type}.png", bbox_inches="tight")
     
    # Plot dos chutes do Kobe (dispersão melhorada)
    plt.figure(figsize=(10, 8))
    
    # Separar acertos e erros
    made_shots = shots_data[shots_data['shot_made_flag'] == 1]
    missed_shots = shots_data[shots_data['shot_made_flag'] == 0]
    
    # Plotar os chutes com bolinhas menores, mais transparentes e com bordas
    plt.scatter(made_shots['loc_x'], made_shots['loc_y'], c='green', label='Acertos', alpha=0.3, s=20, edgecolors='k', linewidth=0.5, zorder=1)
    plt.scatter(missed_shots['loc_x'], missed_shots['loc_y'], c='red', label='Erros', alpha=0.3, s=20, edgecolors='k', linewidth=0.5, zorder=1)
    
    plt.title(f"Chutes do Kobe Bryant ({data_type.capitalize()}) - {model_name}", fontsize=16)
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.legend(loc="upper right")
    plt.xlim(-250, 250)
    plt.ylim(-50, 420)
    plt.grid(True, linestyle='--', alpha=0.7)  # Adiciona um grid pra ajudar na visualização
    plt.savefig(f"data/08_reporting/kobe_shots_{model_name}_{data_type}.png", bbox_inches="tight")
    plt.close()
    print(f"Gráfico de chutes do Kobe salvo: kobe_shots_{model_name}_{data_type}.png")

def get_model_short_name(model):
    # Dicionário mapeando o nome completo para a sigla
    model_name_map = {
        "LogisticRegression": "LR",
        "DecisionTreeClassifier": "DT", 
    }

    # Pega o nome completo da classe do modelo
    full_model_name = type(model).__name__
    
    # Retorna a sigla correspondente ou o nome completo se não encontrado
    return model_name_map.get(full_model_name, full_model_name)

def serve_and_predict(data: pd.DataFrame, run_id: str, model_name: str, port: int = 5001) -> pd.DataFrame:
    """
    Sobe o servidor do MLflow, faz a requisição pra API e retorna as previsões como DataFrame.
    
    Args:
        data: DataFrame com os dados de produção (data_features_prod_processed)
        run_id: ID do run do MLflow onde o modelo foi logado
        model_name: Nome do modelo no MLflow (ex.: 'logistic_regression_model')
        port: Porta onde o servidor do MLflow vai rodar
    
    Returns:
        DataFrame com a coluna 'shot_made_flag_pred' contendo as previsões
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

    # Montar o comando pra subir o servidor do MLflow
    mlflow_command = [
        "mlflow", "models", "serve",
        "-m", f"runs:/{run_id}/{model_name}",
        "-p", str(port),
        "--env-manager", "conda"
    ]

    # Iniciar o servidor do MLflow em background
    print("Iniciando o servidor do MLflow...")
    process = subprocess.Popen(
        mlflow_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Esperar o servidor subir (pode levar alguns segundos)
    time.sleep(10)  # Ajuste o tempo conforme necessário

    # Verificar se o servidor tá rodando
    server_running = False
    for proc in psutil.process_iter(['pid', 'name']):
        if 'mlflow' in proc.info['name'].lower() and proc.pid == process.pid:
            server_running = True
            break

    if not server_running:
        print("Erro ao iniciar o servidor do MLflow:")
        stdout, stderr = process.communicate()
        print("stdout:", stdout)
        print("stderr:", stderr)
        raise RuntimeError("Não foi possível iniciar o servidor do MLflow.")

    try:
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

    finally:
        # Desligar o servidor do MLflow
        print("Desligando o servidor do MLflow...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("Servidor do MLflow desligado.")



def plot_shot_predictions(data: pd.DataFrame, predictions: pd.DataFrame, output_path: str) -> None:
    """
    Gera um gráfico de dispersão mostrando os locais dos arremessos e as previsões.
    Usa bolinhas verdes pra acertos (1) e vermelhas pra erros (0).
    
    Args:
        data: DataFrame com os dados de produção (data_features_prod_processed)
        predictions: DataFrame com as previsões (shot_made_flag_pred)
        output_path: Caminho onde o gráfico será salvo (ex.: 'data/08_reporting/shot_predictions.png')
    
    Returns:
        None
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