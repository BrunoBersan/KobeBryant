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
            inputs=['data_features_prod', 'params:model_name_dt', 'dt_tuned'],
            outputs='predictions_prod_dt',
            name="serve_and_predict_node_dt"
        ),
        node(
            func=plot_shot_predictions_and_metrics,
            inputs=['data_features_prod', 'predictions_prod_dt', 'params:plot_output_path_dt', 'dt_tuned','params:model_name_dt'],
            outputs={
                'metrics': 'metrics_prod_dt',   
                'predicted_probabilities' : 'predictions_proba_prod_dt'
            },
            name="plot_shot_predictions_node_dt"
        )

    ])
