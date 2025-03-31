import pandas as pd
from sklearn.model_selection import train_test_split

def analyze_and_select_features(data: pd.DataFrame) -> pd.DataFrame:  
    """Análise e seleção das features.
    
    Codifica as variáveis categóricas utilizando pd.get_dummies, 
    removendo a primeira coluna para evitar multicolinearidade, 
    e exclui as colunas originais que foram codificadas.
    """    

    features = data[[ 'lat', 'lon', 'minutes_remaining', 'period',
                      'playoffs', 'shot_distance', 
                      'loc_x', 'loc_y', 'shot_made_flag']]

    return features


def split_data(df: pd.DataFrame) -> dict:
    """
    Função para dividir os dados em treino e teste estratificados. proporção 80%/20%    
    """
    X = df.drop(['shot_made_flag'], axis=1)  
    y = df['shot_made_flag']  
     
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
     
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    return {"train": train_df, "test": test_df}