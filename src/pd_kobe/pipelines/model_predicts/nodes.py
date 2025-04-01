"""
Nós criados para analisar e calcular as métricas dos modelos 
e também fazer as predições com os dados de teste do catalog
"""  
import pandas as pd 
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, f1_score, precision_score, recall_score, roc_auc_score, log_loss   
import matplotlib.pyplot as plt
import mlflow
import numpy as np

def calculate_model_metrics(features: pd.DataFrame, session_id, model, str):
    
    # Separar variáveis preditoras e alvo
    X_test = features.drop(['shot_made_flag'], axis=1)
    y_test = features['shot_made_flag']
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] 
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) 
    logLoss = log_loss(y_test, y_proba) 
    
    # Criar um DataFrame único para as métricas
    metrics_df = pd.DataFrame({
        "accuracy": [accuracy],
        "precision": [precision],
        "recall": [recall],
        "f1_score": [f1],
        "roc_auc": [roc_auc],
        "log_loss": [logLoss]
    })
  
    y_proba_df = pd.DataFrame(y_proba, columns=["probability"])
    predictions_df = pd.DataFrame({
        "actual": y_test,
        "predicted": y_pred
    })

    # Retornar os DataFrames como datasets
    return {
        "metrics": metrics_df,
        "predictions": predictions_df,
        "predicted_probabilities": y_proba_df
    }
    
