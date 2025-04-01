import pandas as pd
from pycaret.classification import *
from sklearn.metrics import log_loss
from skopt.space import Real, Categorical, Integer
import mlflow


def configure_pycaret_setup(train_features: pd.DataFrame, session_id) -> ClassificationExperiment:
    exp = ClassificationExperiment()
    exp.setup(
        data=train_features,
        target='shot_made_flag',
        n_jobs=-1,
        use_gpu=True,
        session_id=session_id
    )
    return exp

def logistic_regression_model(train_features: pd.DataFrame, session_id) -> dict:
    # Definir o tracking URI (ajuste conforme sua preferência)
    mlflow.set_tracking_uri("file:///C:/Projetos/especializacao_ia/segundo_modulo/pd-kobe/mlruns")
    while mlflow.active_run():
        mlflow.end_run()

    exp = configure_pycaret_setup(train_features, session_id)
    with mlflow.start_run(run_name="logistic_regression", nested=True):  # Usar execução aninhada
        lr = exp.create_model('lr', verbose=False)
        lr_search_space = {
            'penalty': Categorical(['l1', 'l2', 'elasticnet']),
            'C': Real(0.001, 100, prior='log-uniform'),
            'class_weight': Categorical(['balanced', None]),
            'max_iter': Integer(100, 1000),
            'tol': Real(1e-4, 1e-2, prior='log-uniform'),
            'solver': Categorical(['liblinear', 'saga']),
        }
        if 'elasticnet' in lr_search_space['penalty'].categories:
            lr_search_space['l1_ratio'] = Real(0.1, 0.9)
            lr_search_space['solver'] = Categorical(['saga'])

        mlflow.log_params(lr_search_space)
        tuned_lr = exp.tune_model(
            lr,
            custom_grid=lr_search_space,
            n_iter=100,
            optimize='F1',
            search_library='scikit-optimize',
            search_algorithm='bayesian',
            choose_better=True,
            early_stopping=True,
            early_stopping_max_iters=10,
            verbose=False
        )

        # Obter as probabilidades preditas
        X_test = train_features.drop(columns=['shot_made_flag'])  
        y_test = train_features['shot_made_flag']
        
        y_pred_proba = tuned_lr.predict_proba(X_test)[:, 1] 

        # Calcular o Log Loss
        logloss = log_loss(y_test, y_pred_proba)

        metrics = metrics = exp.pull() 
       # Criar dicionário de métricas
        metrics_dict = {
            "accuracy": metrics.iloc[0]["Accuracy"],
            "precision": metrics.iloc[0]["Prec."],
            "recall": metrics.iloc[0]["Recall"],
            "f1_score": metrics.iloc[0]["F1"],
            "roc_auc": metrics.iloc[0]["AUC"], 
            "kappa": metrics.iloc[0]["Kappa"],
            "mcc": metrics.iloc[0]["MCC"],
            "logloss" : logloss          
        }

        mlflow.log_metric("accuracy", metrics_dict['accuracy'])
        mlflow.log_metric("precision", metrics_dict['precision'])
        mlflow.log_metric("recall", metrics_dict['recall'])
        mlflow.log_metric("f1_score", metrics_dict['f1_score'])
        mlflow.log_metric("roc_auc", metrics_dict['roc_auc'])
        mlflow.log_metric("kappa", metrics_dict['kappa'])
        mlflow.log_metric("mcc", metrics_dict['mcc'])
        mlflow.log_metric("logloss", metrics_dict['logloss'])

        mlflow.sklearn.log_model(tuned_lr, "logistic_regression_model")

    return tuned_lr

def decision_tree_model(train_features: pd.DataFrame, session_id) -> dict:
    # Definir o tracking URI (ajuste conforme sua preferência)
    mlflow.set_tracking_uri("file:///C:/Projetos/especializacao_ia/segundo_modulo/pd-kobe/mlruns")
    while mlflow.active_run():
        mlflow.end_run()

    exp = configure_pycaret_setup(train_features, session_id)
    with mlflow.start_run(run_name="decision_tree", nested=True):  # Usar execução aninhada
        dt = exp.create_model('dt', verbose=False)
        dt_search_space = {
            'criterion': Categorical(['gini', 'entropy']),
            'splitter': Categorical(['best']),
            'max_depth': Integer(3, 20),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 5),
            'max_features': Categorical(['sqrt', 'log2', 0.5]),
            'ccp_alpha': Real(0.0, 0.1),
            'max_leaf_nodes': Integer(10, 50),
        }
        mlflow.log_params(dt_search_space)
        tuned_dt = exp.tune_model(
            dt,
            n_iter=50,
            optimize='F1',
            search_library='scikit-optimize',
            search_algorithm='bayesian',
            early_stopping=True,
            early_stopping_max_iters=10,
            custom_grid=dt_search_space
        ) 

        # Obter as probabilidades preditas
        X_test = train_features.drop(columns=['shot_made_flag'])  
        y_test = train_features['shot_made_flag']
        
        y_pred_proba = tuned_dt.predict_proba(X_test)[:, 1] 

        # Calcular o Log Loss
        logloss = log_loss(y_test, y_pred_proba)

        metrics = metrics = exp.pull() 
       # Criar dicionário de métricas
        metrics_dict = {
            "accuracy": metrics.iloc[0]["Accuracy"],
            "precision": metrics.iloc[0]["Prec."],
            "recall": metrics.iloc[0]["Recall"],
            "f1_score": metrics.iloc[0]["F1"],
            "roc_auc": metrics.iloc[0]["AUC"], 
            "kappa": metrics.iloc[0]["Kappa"],
            "mcc": metrics.iloc[0]["MCC"],
            "logloss" : logloss          
        }

        mlflow.log_metric("accuracy", metrics_dict['accuracy'])
        mlflow.log_metric("precision", metrics_dict['precision'])
        mlflow.log_metric("recall", metrics_dict['recall'])
        mlflow.log_metric("f1_score", metrics_dict['f1_score'])
        mlflow.log_metric("roc_auc", metrics_dict['roc_auc'])
        mlflow.log_metric("kappa", metrics_dict['kappa'])
        mlflow.log_metric("mcc", metrics_dict['mcc'])
        mlflow.log_metric("logloss", metrics_dict['logloss'])

        mlflow.sklearn.log_model(tuned_dt, "decision_tree_model")

    return tuned_dt 