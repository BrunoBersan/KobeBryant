"""Project pipelines."""
from kedro.framework.project import find_pipelines
from pd_kobe.pipelines import data_preparation, data_processing, model_training, model_predicts, reporting, predict_api_decision_tree, predict_api_logistic_regression
from kedro.pipeline import Pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines in execution order.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {
        "data_preparation": data_preparation.create_pipeline(),
        "data_processing": data_processing.create_pipeline(),
        "model_training": model_training.create_pipeline(),
        "model_predicts": model_predicts.create_pipeline(),
        "reporting": reporting.create_pipeline(),
        
        "predict_api_decision_tree": predict_api_decision_tree.create_pipeline(),
        "predict_api_logistic_regression": predict_api_logistic_regression.create_pipeline(),

        "__default__": 
            data_preparation.create_pipeline() + 
            data_processing.create_pipeline() + 
            model_training.create_pipeline() + 
            model_predicts.create_pipeline() + 
            reporting.create_pipeline()
    } 

    return pipelines