"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines in execution order.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()

    # Definir a ordem dos pipelines
    ordered_pipeline = (
        pipelines["data_preparation"] +
        pipelines["data_processing"] +
        pipelines["model_training"] +
        pipelines["model_predicts"] +
        pipelines["reporting"]
    )

    pipelines["__default__"] = ordered_pipeline
    return pipelines
