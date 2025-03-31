# Projeto de Previsão de Arremessos do Kobe Bryant

## Visão Geral

Este projeto foi desenvolvido como parte do **Curso de Especialização em Inteligência Artificial (Segundo Módulo)** e tem como objetivo prever se um arremesso do Kobe Bryant foi convertido ou não com base em dados históricos de jogos da NBA. Utilizando técnicas de machine learning, o projeto emprega o framework **Kedro** para gerenciamento de pipelines de dados, o **PyCaret** para treinamento e ajuste de modelos, e o **MLflow** para rastreamento de experimentos e implantação de modelos. O objetivo é construir um pipeline robusto que processe os dados, treine modelos, faça previsões e visualize os resultados de forma clara e informativa.

O conjunto de dados utilizado contém características como a localização do arremesso (`lat` e `lon`), tipo de arremesso (`shot_type`), tipo de ação (`action_type`), distância do arremesso (`shot_distance`), período do jogo (`period`) e a variável alvo `shot_made_flag` (1 para arremesso convertido, 0 para arremesso errado). O projeto abrange pré-processamento de dados, treinamento de modelos, previsão por meio de uma API e visualização das previsões em um gráfico de dispersão.

---

## Estrutura do Projeto

O projeto é organizado utilizando o framework **Kedro**, que garante um pipeline de dados modular e reprodutível. Abaixo está a estrutura do projeto:

```

pd-kobe/
├── conf/                           # Arquivos de configuração (configurações do Kedro, parâmetros, catálogo)
├── data/                           # Diretórios de dados (brutos, processados, saídas de modelo, etc.)
│   ├── 01_raw/                     # Conjunto de dados bruto
│   ├── 02_intermediate/            # Dados intermediários
│   ├── 03_primary/                 # Dados primários
│   ├── 04_feature/                 # Dados com engenharia de features
│   ├── 05_model_input/             # Dados prontos para o modelo
│   ├── 06_models/                  # Modelos treinados
│   ├── 07_model_output/            # Saídas do modelo (previsões)
│   ├── 08_reporting/               # Visualizações e relatórios (gráficos)
├── src/                            # Código-fonte do projeto
│   ├── pd_kobe/                    # Módulo principal do projeto
│   │   ├── pipelines/              # Pipelines do Kedro
│   │   │   ├── data_preparation/   # Pipeline de pré-processamento
│   │   │   ├── data_processing/    # Pipeline de seleção e engenharia de features
│   │   │   ├── model_training/     # Pipeline de treinamento de modelos
│   │   │   ├── model_predicts/     # Pipeline de resultados e previsões dos modelos treinados
│   │   │   ├── reporting/          # Pipeline de gráficos e métricas visuais
│   │   │   └── nodes.py            # Funções dos nós do pipeline
│   │   ├── settings.py             # Configurações do projeto
│   │   └── __init__.py             # Arquivo de inicialização
├── mlruns/                         # Diretório do MLflow para rastreamento de experimentos
├── predict.py                      # Script para previsões manuais (opcional)
├── requirements.txt                # Dependências do projeto
└── README.md                       # Documentação do projeto

```
---

## Funcionalidades Principais

1. **Pré-processamento de Dados**:
   - 

2. **Treinamento de Modelos**:
   - Dois modelos são treinados: Regressão Logística (`logistic_regression_model`) e Árvore de Decisão (`decision_tree_model`).
   - O PyCaret é usado para configurar o ambiente de treinamento, criar os modelos e ajustá-los com busca bayesiana de hiperparâmetros (usando a biblioteca `scikit-optimize`).
   - O MLflow rastreia os experimentos, logando métricas e os modelos treinados.

3. **Previsão via API**:
   - O modelo treinado é servido via MLflow usando o comando `mlflow models serve`.
   - Um nó do pipeline (`serve_and_predict`) automatiza o processo de subir o servidor do MLflow, fazer a requisição à API (`/invocations`) e desligar o servidor.
   - As previsões são salvas em `data/07_model_output/predictions.csv`.

4. **Visualização**:
   - Um gráfico de dispersão é gerado para visualizar os locais dos arremessos (`lat` e `lon`) e as previsões do modelo.
   - Arremessos previstos como convertidos (`1`) são representados por **bolinhas verdes**, e arremessos previstos como errados (`0`) por **bolinhas vermelhas**.
   - O gráfico é salvo em `data/08_reporting/shot_predictions.png`.

---

## Pré-requisitos

Para executar o projeto, você precisa ter as seguintes ferramentas instaladas:

- **Python 3.8+**
- **Conda** (para gerenciamento de ambientes)
- **Kedro 0.19.12**
- **MLflow**
- **PyCaret**
- **Pandas**
- **Matplotlib** (para visualizações)
- **Seaborn** (opcional, para gráficos estilizados)
- **Scikit-learn**
- **Scikit-optimize** (para ajuste de hiperparâmetros)

Você pode instalar as dependências listadas no arquivo `requirements.txt`:

pip install -r requirements.txt

## Como executar o projeto

1. **Configurar o ambiente**
    - Crie e ative um ambiente Conda para o projeto:
        conda create -n kedro_env python=3.8
        conda activate kedro_env
        pip install -r requirements.txt

2. **Configurar o MLflow**
    - Defina o URI de rastreamento do MLflow (já configurado no projeto):
        export MLFLOW_TRACKING_URI=file:///C:/Projetos/especializacao_ia/segundo_modulo/pd-kobe/mlruns

3. **Executar o Pipeline**
    - kedro run

    Isso irá:
    1. Pré-processar os dados.
    2. Treinar os modelos (Regressão Logística e Árvore de Decisão).
    3. Fazer previsões dos modelos localmente e também usando a API do MLflow.
    4. Gerar o gráfico de dispersão, curva, roc, métricas e mais.


## Detalhes dos Pipelines

1. **Preparação dos Dados (data_preparation)**
O pipeline data_preparation é responsável pela limpeza inicial dos dados brutos, garantindo que estejam prontos para as próximas etapas do projeto. Ele processa tanto o conjunto de dados principal (data_shots) quanto o conjunto de produção (data_shots_prod). As principais etapas incluem:

- Tratamento de Valores Nulos (handle_missing_values): Remove todas as linhas que contêm valores nulos no conjunto de dados, garantindo que o modelo receba apenas dados completos.

- Remoção de Duplicatas e Validações (remove_duplicates_and_validate): Elimina registros duplicados, mantendo apenas a última ocorrência, e realiza validações adicionais para assegurar a integridade dos dados.


2. **Processamento dos Dados e Seleção de Features (data_processing)**
O pipeline data_processing realiza o processamento e a seleção de features, preparando os dados para o treinamento do modelo. Ele também divide o conjunto de dados em treino e teste. As etapas incluem:

- **Análise e Seleção de Features (analyze_and_select_features)**: Seleciona um subconjunto de features relevantes para o modelo, incluindo lat, lon, minutes_remaining, period, playoffs, shot_distance, loc_x, loc_y e shot_made_flag. Essa etapa é aplicada tanto ao conjunto de dados principal (data_shots_normalized) quanto ao conjunto de produção (data_shots_prod_normalized), gerando os datasets data_features e data_features_prod, respectivamente.

- **Divisão dos Dados (split_data)**: Divide o conjunto de dados data_features em conjuntos de treino (shots_train) e teste (shots_test) na proporção 80/20, utilizando estratificação com base na variável alvo shot_made_flag para manter a proporção de classes. A divisão é feita com um random_state=42 para garantir reprodutibilidade.


- **split_data_node**: Divide o conjunto de dados principal em treino e teste.

3. **Treinamento de Modelos (model_training)**
O pipeline model_training é responsável pelo treinamento e ajuste de dois modelos de machine learning: uma Regressão Logística e uma Árvore de Decisão. Ele utiliza o PyCaret para configurar o ambiente de treinamento e o MLflow para rastrear os experimentos. As etapas incluem:

- **Configuração do PyCaret (configure_pycaret_setup)**: Configura o ambiente de treinamento do PyCaret, definindo a variável alvo (shot_made_flag), utilizando todos os núcleos disponíveis (n_jobs=-1) e habilitando o uso de GPU (use_gpu=True).
- **Treinamento da Regressão Logística (logistic_regression_model)**: Treina um modelo de Regressão Logística com ajuste de hiperparâmetros usando busca bayesiana (via scikit-optimize). O espaço de busca inclui parâmetros como penalty, C, class_weight, max_iter, tol e solver. O modelo é otimizado com base no F1 Score, e os resultados (métricas e modelo) são logados no MLflow.
- **Treinamento da Árvore de Decisão (decision_tree_model)**: Treina um modelo de Árvore de Decisão com ajuste de hiperparâmetros, também usando busca bayesiana. O espaço de busca inclui parâmetros como criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_features, ccp_alpha e max_leaf_nodes. O modelo é otimizado com base no F1 Score, e os resultados são logados no MLflow.
O pipeline contém dois nós:

- *logistic_regression_model*: Treina e ajusta a Regressão Logística, gerando o modelo lr_tuned.
- *decision_tree_model*: Treina e ajusta a Árvore de Decisão, gerando o modelo dt_tuned.

4. **Previsões dos Modelos com Dados de Treino e Teste (model_predicts)**
O pipeline model_predicts realiza previsões com os modelos treinados (Regressão Logística e Árvore de Decisão) nos conjuntos de treino (shots_train) e teste (shots_test), além de calcular métricas de desempenho. As etapas incluem:

- **Cálculo de Métricas e Previsões (calculate_model_metrics)**: Faz previsões com os modelos nos dados de treino e teste, calcula métricas de desempenho (acurácia, precisão, recall, F1 Score e ROC AUC) e retorna três datasets:
- Um DataFrame com as métricas (metrics).
- Um DataFrame com as previsões e os valores reais (predictions).
- Um DataFrame com as probabilidades previstas (predicted_probabilities).


5. **Relatórios, Gráficos e Métricas (reporting)**
O pipeline reporting é responsável por gerar relatórios visuais e gráficos para avaliar o desempenho dos modelos e visualizar os resultados das previsões. Ele também faz previsões no conjunto de produção (data_features_prod) via API do MLflow. As etapas incluem:

- **Geração de Relatórios Visuais (save_model_plots_metrics)**: Gera cinco tipos de visualizações para cada modelo (Regressão Logística e Árvore de Decisão) nos conjuntos de treino e teste:

- **Matriz de Confusão**: Mostra a distribuição de previsões corretas e incorretas.
- **Curva ROC**: Exibe a curva ROC e o valor de AUC para avaliar a capacidade de discriminação do modelo.
- **Tabela de Métricas**: Apresenta as métricas de desempenho (acurácia, precisão, recall, F1 Score, ROC AUC) em formato de tabela.
- **Distribuição de Probabilidades**: Plota um histograma das probabilidades previstas, mostrando a distribuição das previsões.
- **Gráfico de Chutes do Kobe**: Um gráfico de dispersão que mostra os locais dos arremessos (loc_x e loc_y), com bolinhas verdes para acertos e vermelhas para erros, com base nos valores reais (shot_made_flag).
Esses gráficos são salvos no diretório data/08_reporting/ com nomes que indicam o modelo e o conjunto de dados (ex.: confusion_matrix_report_LR_train.png).
- **Previsão via API (serve_and_predict)**: Sobe o servidor do MLflow, faz a requisição à API (/invocations) para prever os arremessos no conjunto de produção (data_features_prod) e retorna as previsões como um DataFrame (predictions).
- **Gráfico de Previsões (plot_shot_predictions)**: Gera um gráfico de dispersão com os locais dos arremessos no conjunto de produção (lat e lon), usando as previsões do modelo. Arremessos previstos como convertidos (1) são representados por bolinhas verdes, e arremessos previstos como errados (0) por bolinhas vermelhas. O gráfico é salvo em data/08_reporting/shot_predictions.png.


## Resultados

**Modelos Treinados**: Os modelos são salvos no MLflow e podem ser acessados via UI (mlflow ui) em http://localhost:5000.

**Previsões**: As previsões são binárias (0 para erro, 1 para acerto) e salvas em um arquivo CSV.

**Gráfico de Dispersão**: O gráfico mostra os locais dos arremessos com bolinhas verdes (acertos) e vermelhas (erros), facilitando a análise visual das previsões.


## Possíveis Melhorias

**Obter Probabilidades**: Atualmente, a API do MLflow retorna apenas previsões binárias. Uma melhoria seria ajustar o modelo servido para retornar as probabilidades (predict_proba) e usá-las para colorir o gráfico com um gradiente.

**Balanceamento de Classes**: Se o modelo prever apenas uma classe (ex.: todos 0), pode ser necessário balancear o conjunto de dados ou ajustar os pesos das classes no treinamento.

**Mais Features**: Adicionar novas features (ex.: interações entre lat e lon, ou shot_distance ao quadrado) pode melhorar o desempenho do modelo.

**Modelos Mais Complexos**: Testar modelos mais avançados, como Random Forest ou Gradient Boosting, pode capturar melhor os padrões nos dados.


## Sinta-se à vontade para usar, modificar e distribuir o código conforme necessário. ##