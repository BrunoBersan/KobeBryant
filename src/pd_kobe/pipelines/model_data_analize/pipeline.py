from kedro.pipeline import Pipeline, node
from .nodes import label_dataset, combine_datasets, split_train_test, check_separability, check_data_drift

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # Rotula o dataset de produção com data_new=1
            node(
                func=label_dataset,
                inputs=['data_features_prod', 'params:data_processing.label_producao'],
                outputs="producao_data_labeled",
                name="label_producao_data_node",
            ),
            # Rotula o dataset de homologação com data_new=0
            node(
                func=label_dataset,
                inputs=['data_features', 'params:data_processing.label_homologacao'],
                outputs="homologacao_data_labeled",
                name="label_homologacao_data_node",
            ),
            # Combina os dois datasets
            node(
                func=combine_datasets,
                inputs=["producao_data_labeled", "homologacao_data_labeled"],
                outputs="combined_data_analize",
                name="combine_datasets_node",
            ),
            # Separa em treino e teste
            node(
                func=split_train_test,
                inputs="combined_data_analize",
                outputs=["train_data_analize", "test_data_analize"],
                name="split_train_test_node",
            ),
            # Verifica se os dados são separáveis
            node(
                func=check_separability,
                inputs=["train_data_analize", "test_data_analize"],
                outputs="separability_report",
                name="check_separability_node",
            ),
            # Verifica data drift
            node(
                func=check_data_drift,
                inputs=["producao_data_labeled", "homologacao_data_labeled"],
                outputs="drift_report",
                name="check_data_drift_node",
            ),
        ]
    )