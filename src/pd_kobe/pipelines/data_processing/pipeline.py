"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import analyze_and_select_features, split_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([                        
        node(func=analyze_and_select_features, 
            inputs="data_shots_normalized", 
            outputs="data_features", 
            name="analyze_and_select_features_node"),

        node(func=analyze_and_select_features, 
            inputs="data_shots_prod_normalized", 
            outputs="data_features_prod", 
            name="analyze_and_select_features_node_prod"),

        node(func=split_data, 
             inputs="data_features", 
             outputs={"train": "shots_train", "test": "shots_test"}, 
             name="split_data_node"),
    ])
