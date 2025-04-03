"""
This is a boilerplate pipeline 'model_serving'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import serve_and_predict, plot_shot_predictions_and_metrics

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=serve_and_predict,
            inputs=['data_features_prod','params:model_name_lr', 'lr_tuned'],
            outputs='predictions_prod_lr',
            name="serve_and_predict_node_lr"
        ),
        node(
            func=plot_shot_predictions_and_metrics,
            inputs=['data_features_prod', 'predictions_prod_lr', 'params:plot_output_path_lr', 'lr_tuned', 'params:model_name_lr'],
            outputs={
                'metrics': 'metrics_prod_lr',  
                'predicted_probabilities' : 'predictions_proba_prod_lr'
            },
            name="plot_shot_predictions_node_lr"
        ),

    ])
