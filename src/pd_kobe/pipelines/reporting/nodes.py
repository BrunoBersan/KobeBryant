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

