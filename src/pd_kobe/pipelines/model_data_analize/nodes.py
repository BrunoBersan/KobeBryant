from datetime import date
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import logging
import mlflow

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def label_dataset(dataset: pd.DataFrame, label: int) -> pd.DataFrame:
    """Adiciona a coluna 'data_new' com o valor especificado (1 ou 0)."""
    dataset_labeled = dataset.copy()
    dataset_labeled["data_new"] = label
    logger.info(f"Dataset rotulado com data_new={label}, shape: {dataset_labeled.shape}")
    return dataset_labeled

def combine_datasets(producao_data: pd.DataFrame, homologacao_data: pd.DataFrame) -> pd.DataFrame:
    """Combina os dois datasets em um único DataFrame."""
    combined = pd.concat([producao_data, homologacao_data], ignore_index=True)
    logger.info(f"Datasets combinados, shape: {combined.shape}")
    return combined

def split_train_test(combined_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa os dados em treino e teste (80/20) com estratificação pela coluna 'data_new'."""
    train_data, test_data = train_test_split(
        combined_data,
        test_size=0.2,
        stratify=combined_data["data_new"],
        random_state=42
    )
    logger.info(f"Treino: {train_data.shape}, Teste: {test_data.shape}")
    return train_data, test_data

def check_separability(train_data: pd.DataFrame, test_data: pd.DataFrame) -> dict:
    """Verifica se os dados são separáveis, gera gráficos de métricas, curva ROC e dispersão."""
    # Separa features e target
    X_train = train_data.drop(columns=["data_new"])
    y_train = train_data["data_new"]
    X_test = test_data.drop(columns=["data_new"])
    y_test = test_data["data_new"]

    # Treina um modelo simples
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Avalia o modelo
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info("Relatório de separabilidade gerado.")

    # Prepara os dados para o gráfico de métricas
    metrics = ["precision", "recall", "f1-score"]
    classes = ["0", "1"]
    data = {
        "Metric": [],
        "Value": [],
        "Class": []
    }
    for cls in classes:
        for metric in metrics:
            data["Metric"].append(metric)
            data["Value"].append(report[cls][metric])
            data["Class"].append(f"data_new={cls}")

    df_metrics = pd.DataFrame(data)

    # Gráfico de métricas
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Value", y="Metric", hue="Class", data=df_metrics)
    plt.title("Separability Metrics: Produção vs Homologação")
    plt.xlabel("Value")
    plt.ylabel("Metric")
    plt.tight_layout()
    model_name = "logistic_regression"
    data_type = "test"
    metrics_plot_path = f"data/08_reporting/separability_metrics_{model_name}_{data_type}.png"
    plt.savefig(metrics_plot_path)
    plt.close()

    # Gráfico de curva ROC
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Produção vs Homologação")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_plot_path = f"data/08_reporting/roc_curve_report_{model_name}_{data_type}.png"
    plt.savefig(roc_plot_path)
    plt.close()

    # Gráfico de dispersão
    # Seleciona as duas primeiras features para o scatter plot
    if X_test.shape[1] >= 2:  # Verifica se há pelo menos 2 features
        feature1, feature2 = X_test.columns[:2]  # Pega as duas primeiras features
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            X_test[feature1],
            X_test[feature2],
            c=y_test,
            cmap="coolwarm",
            alpha=0.6,
            edgecolors="w",
            s=100,
            label=y_test
        )
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title("Scatter Plot: Produção vs Homologação")
        plt.legend(handles=scatter.legend_elements()[0], labels=["Homologação (0)", "Produção (1)"])
        plt.tight_layout()
        scatter_plot_path = f"data/08_reporting/scatter_plot_{model_name}_{data_type}.png"
        plt.savefig(scatter_plot_path)
        plt.close()
    else:
        logger.warning("Não há features suficientes para criar um gráfico de dispersão (mínimo 2 features).")

    # Loga no MLflow
    run_name = f"check_separability_{date.today().strftime('%Y-%m-%d')}"
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.set_tag("project_name", "projeto_kobe")
        mlflow.set_tag("stage", "check_separability")    
        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.log_metric("f1_score_class_0", report["0"]["f1-score"])
        mlflow.log_metric("f1_score_class_1", report["1"]["f1-score"])
        mlflow.log_artifact(metrics_plot_path)
        mlflow.log_artifact(roc_plot_path)
        if X_test.shape[1] >= 2:  # Só loga o scatter plot se ele foi criado
            mlflow.log_artifact(scatter_plot_path)

    logger.info("Gráficos de separabilidade gerados.")
    return report

def check_data_drift(producao_data: pd.DataFrame, homologacao_data: pd.DataFrame) -> dict:
    """Verifica se há data drift e gera um relatório HTML."""
    # Remove a coluna 'data_new' para comparar apenas as features originais
    producao_data = producao_data.drop(columns=["data_new"])
    homologacao_data = homologacao_data.drop(columns=["data_new"])

    # Cria o relatório de data drift usando Evidently
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=homologacao_data, current_data=producao_data)
    drift_dict = drift_report.as_dict()

    # Salva o relatório HTML diretamente
    drift_report_path = "data/08_reporting/drift_report.html"
    drift_report.save_html(drift_report_path)

    # Loga no MLflow
      # Loga no MLflow
    run_name = f"check_data_drift_{date.today().strftime('%Y-%m-%d')}"
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.set_tag("project_name", "projeto_kobe")
        mlflow.set_tag("stage", "check_separability")    
        mlflow.log_artifact(drift_report_path)

    logger.info("Relatório de data drift gerado.")
    return drift_dict