from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import calculate_model_metrics
import pandas as pd

def generate_node(inputDataset, inputModel, inputDataSetStr, outMetric, outPredict, outProba, strName):
    return node(
            func=calculate_model_metrics,
            inputs=[inputDataset, 'params:session_id', inputModel, inputDataSetStr],
            outputs={
                'metrics': outMetric,  
                'predictions' : outPredict ,
                'predicted_probabilities' : outProba
            },
            name=strName
        )

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # TRAIN
        generate_node('shots_train', 'lr_tuned', 'params:data_shots_train_str', 'metrics_lr_train', 'predictions_lr_train', 'predicted_probabilities_lr_train','calculate_model_metrics_node_LR_train'),
        generate_node('shots_train', 'dt_tuned', 'params:data_shots_train_str', 'metrics_dt_train', 'predictions_dt_train', 'predicted_probabilities_dt_train','calculate_model_metrics_node_DT_train'),
        
        # TEST 
        generate_node('shots_test', 'lr_tuned', 'params:data_shots_test_str', 'metrics_lr_test', 'predictions_lr_test', 'predicted_probabilities_lr_test','calculate_model_metrics_node_LR_test'),
        generate_node('shots_test', 'dt_tuned', 'params:data_shots_test_str', 'metrics_dt_test', 'predictions_dt_test', 'predicted_probabilities_dt_test','calculate_model_metrics_node_DT_test'),
    ])
