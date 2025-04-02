import streamlit as st
import pickle
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

@st.cache_resource
def load_model(model_name):
    model_path = f'../data/06_models/{model_name}.pickle'
    
    if not os.path.exists(model_path):
        st.error(f"Arquivo {model_path} não encontrado.")
        return None
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

st.write("""
    # 🎯 Inferência dos dados  

    Informe os dados para realizar uma previsão 
""")

# Carregar os modelos
lr_model: LogisticRegression = load_model('lr_tuned')
dt_model: DecisionTreeClassifier = load_model('dt_tuned')

# Verifica se os modelos foram carregados antes de usá-los
if lr_model and dt_model:
    models = [lr_model, dt_model] 
else:
    st.warning("Algum modelo não foi carregado corretamente.")

# Seletor de modelo
model_choice = st.pills(
    "Modelos:", 
    ["Regressão Logistica", "Árvore de decisão"], 
    selection_mode='single', 
    default=["Regressão Logistica"]
)

# Inputs do usuário
lat = st.number_input("Latitude", key=1)
lon = st.number_input("Longitude", key=2)
minutes = st.number_input("Minutos restantes", key=3)

# Garante que o período sempre tenha um valor padrão
period_str = st.selectbox("Período", ['1°', '2°', '3°', '4°'], index=0, key=4)
period = int(period_str[0]) if period_str else 1  # Se for None, assume '1'

playoffs = st.checkbox("Playoffs?", key=5)
shot_distance = st.number_input("Distância do arremesso", key=6)
loc_x = st.number_input("Localização X na quadra", key=7)
loc_y = st.number_input("Localização Y na quadra", key=8)

# Converter playoffs para 0 ou 1
playoffs = 1 if playoffs else 0  

if st.button("Prever acerto", key=9):
    input_data = np.array([[lat, lon, minutes, period, playoffs, shot_distance, loc_x, loc_y]])

    if model_choice == "Regressão Logistica":
        prediction = lr_model.predict(input_data)
    else:
        prediction = dt_model.predict(input_data)

    # Exibir o resultado
    if prediction[0] == 1:
        st.success(f"Previsão do modelo: 'Acertou' =D")
        st.balloons()
    else:
        st.success(f"Previsão do modelo: 'Errou' =/") 
