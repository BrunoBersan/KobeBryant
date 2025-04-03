"""
This is a boilerplate pipeline 'reporting'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import save_model_plots_metrics

##$metrics: pd.DataFrame, predicted_probabilities: pd.DataFrame, predictions: pd.DataFrame, model

def generate_node(metrics, shots, predictProba, predictions, model, dataShotStr, functionName):
    return node(
            func=save_model_plots_metrics,
            inputs=[metrics,shots, predictProba,predictions, model, dataShotStr],
            outputs=None,
            name=functionName
        )

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        #train
        generate_node('metrics_lr_train','shots_train','predicted_probabilities_lr_train','predictions_lr_train','lr_tuned','params:data_shots_train_str','save_model_plots_metrics_LR_train'),
        generate_node('metrics_dt_train','shots_train','predicted_probabilities_dt_train','predictions_dt_train', 'dt_tuned', 'params:data_shots_train_str','save_model_plots_metrics_DT_train'),

        #test
        generate_node('metrics_lr_test','shots_test','predicted_probabilities_lr_test','predictions_lr_test', 'lr_tuned', 'params:data_shots_test_str','save_model_plots_metrics_LR_test'),
        generate_node('metrics_dt_test','shots_test','predicted_probabilities_dt_test','predictions_dt_test', 'dt_tuned', 'params:data_shots_test_str','save_model_plots_metrics_DT_test'),         
    ])