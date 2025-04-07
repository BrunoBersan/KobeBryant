import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

def analyze_and_select_features(data: pd.DataFrame) -> pd.DataFrame:  
    """Análise e seleção das features.    
    """    

    features = data[[ 'lat', 'lon', 'minutes_remaining', 'period',
                      'playoffs', 'shot_distance', 
                      'loc_x', 'loc_y', 'shot_made_flag']]

    return features


def split_data(df: pd.DataFrame) -> dict:
    """
    Função para dividir os dados em treino e teste estratificados. proporção 80%/20%    
    """
def split_data(df: pd.DataFrame) -> dict:
    """
    Função para dividir os dados em treino e teste estratificados. proporção 80%/20%.
    Registra parâmetros e métricas no MLflow.
    """
    X = df.drop(['shot_made_flag'], axis=1)  
    y = df['shot_made_flag']  
    
    test_size = 0.2

    mlflow.set_experiment("ProjetoKobe")
    with mlflow.start_run(run_name="PreparacaoDados"):
        # Split estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        # Log de parâmetros
        mlflow.log_param("test_size", test_size)

        # Log de métricas
        mlflow.log_metric("train_size", train_df.shape[0])
        mlflow.log_metric("test_size", test_df.shape[0])

        # (Opcional) Log de distribuição das classes
        mlflow.log_metric("train_target_mean", y_train.mean())
        mlflow.log_metric("test_target_mean", y_test.mean())
    
    return {"train": train_df, "test": test_df}