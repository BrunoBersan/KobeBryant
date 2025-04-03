"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import logistic_regression_model, decision_tree_model

def create_pipeline(**kwargs) -> Pipeline: 

    return pipeline([
        node(func=logistic_regression_model, 
            inputs=['shots_train', 'params:session_id', 'params:path_mlflow_runs'], 
            outputs='lr_tuned', 
            name="logistic_regression_model"),

        node(func=decision_tree_model, 
            inputs=['shots_train', 'params:session_id','params:path_mlflow_runs'], 
            outputs='dt_tuned', 
            name="decision_tree_model"),        

    ])