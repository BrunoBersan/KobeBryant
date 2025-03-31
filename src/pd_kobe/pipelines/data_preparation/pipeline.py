"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

from .nodes import handle_missing_values, remove_duplicates_and_validate


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
            [
                #test
                node(func=handle_missing_values, 
                    inputs="data_shots", 
                    outputs="data_shots_not_null", 
                    name="handle_missing_values_node"),
                
                node(func=remove_duplicates_and_validate, 
                    inputs="data_shots_not_null", 
                    outputs="data_shots_normalized", 
                    name="remove_duplicates_and_validate_node"),

                #prod
                node(func=handle_missing_values, 
                    inputs="data_shots_prod", 
                    outputs="data_shots_prod_not_null", 
                    name="handle_missing_values_node_prod"),
                
                node(func=remove_duplicates_and_validate, 
                    inputs="data_shots_prod_not_null", 
                    outputs="data_shots_prod_normalized", 
                    name="remove_duplicates_and_validate_node_prod")
            ]
        )
