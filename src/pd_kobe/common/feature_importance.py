from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt

def show_importance_features(model, X_train, model_name, data_type):
    
    try:
        feature_names = feature_names = X_train.columns.tolist()
        feature_names = [name.split("__")[-1] for name in feature_names]  # Remove "num_continuas__"
    except AttributeError:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]  # Nomes genéricos
    # Verificar o tipo do modelo corretamente
    if isinstance(model, (LogisticRegression)):
        if hasattr(model, "coef_"):   
            coef = model.coef_.flatten()  # Achata os coeficientes
            # Verifica se o número de coeficientes corresponde ao número de atributos
            if len(coef) == len(feature_names):
                imp = pd.DataFrame({"atributos": feature_names, "importancia": coef})
                imp = imp.sort_values(by="importancia", ascending=True)
                # Construindo gráfico
                plt.figure(figsize=(10, 6))
                plt.barh(y=imp['atributos'], width=imp['importancia'])
                plt.xlabel("Importância")
                plt.ylabel("Atributos")
                plt.title(f"Importância dos Atributos - {model.__class__.__name__}")
                plt.savefig(f"data/08_reporting/feature_importante_{model_name}_{data_type}.png", bbox_inches="tight")
                plt.close()
            else:
                print(f"Erro: O número de coeficientes ({len(coef)}) não bate com o número de atributos ({len(feature_names)})")
        else:
            print(f"O modelo {model.__class__.__name__} usa um kernel que não fornece coeficientes.")
    elif isinstance(model, DecisionTreeClassifier):
        imp = pd.DataFrame({"atributos": feature_names, "importancia": model.feature_importances_})
        imp = imp.sort_values(by="importancia", ascending=True) 
        # Construindo gráfico
        plt.figure(figsize=(10, 6))
        plt.barh(y=imp['atributos'], width=imp['importancia'])
        plt.xlabel("Importância")
        plt.ylabel("Atributos")
        plt.title(f"Importância dos Atributos - {model.__class__.__name__}")
        plt.savefig(f"data/08_reporting/feature_importante_{model_name}_{data_type}.png", bbox_inches="tight")
        plt.close()
    else:
        print(f"O modelo {model.__class__.__name__} não fornece coeficientes de importância.")