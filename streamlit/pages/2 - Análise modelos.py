import streamlit as st
import pickle
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Função para carregar os modelos
@st.cache_resource
def load_model(model_name):
    model_path = f'../data/06_models/{model_name}.pickle'
    if not os.path.exists(model_path):
        st.error(f"Arquivo {model_path} não encontrado.")
        return None
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# Função para converter classification_report em DataFrame
def report_to_df(report):
    report_dict = {}
    for line in report.split('\n')[2:-4]:  # Pular cabeçalho e rodapé
        if line.strip():  # Ignorar linhas vazias
            row = line.split()
            report_dict[row[0]] = {
                'Precisão': float(row[1]),
                'Recall': float(row[2]),
                'F1-Score': float(row[3]),
                'Suporte': int(row[4])
            }
    return pd.DataFrame(report_dict).T

# Carregar os modelos
lr_model: LogisticRegression = load_model('lr_tuned')
dt_model: DecisionTreeClassifier = load_model('dt_tuned')

# Carregar o dataset real
data_path = '../data/04_feature/data_features_prod.parquet'
if not os.path.exists(data_path):
    st.error(f"Arquivo {data_path} não encontrado.")
else:
    df = pd.read_parquet(data_path)
    # Suposição: 'shot_made_flag' é a variável alvo; ajustar conforme necessário
    y_true = df['shot_made_flag']
    X_test = df.drop(columns=['shot_made_flag'])  # Todas as outras colunas como features

    # Verificar se os modelos foram carregados corretamente
    if lr_model and dt_model:
        # Fazer previsões
        y_pred_lr = lr_model.predict(X_test)
        y_pred_dt = dt_model.predict(X_test)

        # Calcular métricas detalhadas
        metrics = {
            "Modelo": ["Regressão Logística", "Árvore de Decisão"],
            "Acurácia": [accuracy_score(y_true, y_pred_lr), accuracy_score(y_true, y_pred_dt)],
            "Precisão": [precision_score(y_true, y_pred_lr), precision_score(y_true, y_pred_dt)],
            "Recall": [recall_score(y_true, y_pred_lr), recall_score(y_true, y_pred_dt)],
            "F1-Score": [f1_score(y_true, y_pred_lr), f1_score(y_true, y_pred_dt)]
        }
        df_metrics = pd.DataFrame(metrics)

        # Criar página
        st.title("Estatísticas dos Modelos")
        st.write("Comparação entre os modelos treinados para prever os arremessos de Kobe Bryant.")

        # Exibir métricas em uma datagrid estilizada
        st.write("### Comparação de Modelos")
        st.dataframe(
            df_metrics.style
            .format({"Acurácia": "{:.2%}", "Precisão": "{:.2%}", "Recall": "{:.2%}", "F1-Score": "{:.2%}"})
            .background_gradient(cmap='Blues', subset=["Acurácia", "Precisão", "Recall", "F1-Score"])
            .set_properties(**{'text-align': 'center', 'font-size': '14px', 'border': '1px solid #ddd'})
        )

        # Gráficos de matrizes de confusão
        st.write("### Matrizes de Confusão")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        cm_lr = confusion_matrix(y_true, y_pred_lr)
        cm_dt = confusion_matrix(y_true, y_pred_dt)
        
        sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title("Regressão Logística")
        
        sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
        axes[1].set_title("Árvore de Decisão")
        
        st.pyplot(fig)

        # Gráfico comparativo de métricas
        st.write("### Comparação Visual de Métricas")
        fig2, ax = plt.subplots(figsize=(10, 6))
        df_metrics.set_index("Modelo").plot(kind="bar", ax=ax, colormap="viridis")
        plt.title("Desempenho dos Modelos")
        plt.ylabel("Pontuação (%)")
        plt.xticks(rotation=0)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2%}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom')
        st.pyplot(fig2)

        # Exibir relatórios de classificação como tabelas
        st.write("### Relatórios de Classificação")
        
        st.write("#### Regressão Logística")
        report_lr = classification_report(y_true, y_pred_lr)
        df_report_lr = report_to_df(report_lr)
        st.dataframe(
            df_report_lr.style
            .format({"Precisão": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}", "Suporte": "{:.0f}"})
            .background_gradient(cmap='Greens')
            .set_properties(**{'text-align': 'center', 'font-size': '12px', 'border': '1px solid #ddd'})
        )
        
        st.write("#### Árvore de Decisão")
        report_dt = classification_report(y_true, y_pred_dt)
        df_report_dt = report_to_df(report_dt)
        st.dataframe(
            df_report_dt.style
            .format({"Precisão": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}", "Suporte": "{:.0f}"})
            .background_gradient(cmap='Oranges')
            .set_properties(**{'text-align': 'center', 'font-size': '12px', 'border': '1px solid #ddd'})
        )

    else:
        st.warning("Os modelos não foram carregados corretamente. Verifique os arquivos de modelo.")