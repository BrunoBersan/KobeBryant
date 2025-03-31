"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 0.19.12
"""
import pandas as pd

def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame: 
    """Tratamento de valores nulos"""
    return data.dropna()    

def remove_duplicates_and_validate(data: pd.DataFrame) -> pd.DataFrame: 
    """Remoção de duplicatas e validações diversas"""
    return data.drop_duplicates(keep='last')     
