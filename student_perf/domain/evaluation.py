"""
Fonctions d'évaluation des modèles entraînés.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

try:
    import shap
except ImportError:
    shap = None



def evaluation_modele_classifier(y_test: list, y_pred: list) -> tuple[float, float, float, list]:
    """
    Calcule et affiche les métriques d'évaluation d'un modèle classifier.

    Paramètres
    ----------
    y_test : list
        La variable cible réelle (jeu de test).
    y_pred : list
        Les prédictions de classes effectuées par le modèle.

    Retours
    -------
    tuple[float, float, float, list]
        Un tuple contenant l'accuracy, l'AUC, le F1-score et la matrice de confusion.
    """
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    cfm = confusion_matrix(y_test, y_pred)

    print("--- ÉVALUATION DU CLASSIFIEUR ---")
    print("L'accuracy est de :", accuracy)
    print("L'AUC est de :", auc)
    print("Le F1-score est de :", f1)
    print("La matrice de confusion :")
    print(cfm)

    return accuracy, auc, f1, cfm


def evaluation_modele_regressor(
    y_test: list | pd.Series | np.ndarray,
    y_pred: list | pd.Series | np.ndarray,
) -> tuple[float, float, float, float]:
    """Calcule et affiche les métriques d'évaluation d'un modèle de régression.

    Paramètres
    ----------
    y_test : list | pd.Series | np.ndarray
        La variable cible réelle (jeu de test).
    y_pred : list | pd.Series | np.ndarray
        Les prédictions continues effectuées par le modèle.

    Retours
    -------
    tuple[float, float, float, float]
        Un tuple contenant la MAE, la MSE, la RMSE et le score R².
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("--- ÉVALUATION DU RÉGRESSEUR ---")
    print(f"La MAE (Mean Absolute Error) est de : {mae:.4f}")
    print(f"La MSE (Mean Squared Error) est de : {mse:.4f}")
    print(f"La RMSE (Root Mean Squared Error) est de : {rmse:.4f}")
    print(f"Le coefficient de détermination R² est de : {r2:.4f}")

    return mae, mse, rmse, r2


def feature_importances_model(model, X_train: pd.DataFrame, y_train: list | pd.Series) -> pd.DataFrame:
    """Affiche un graphique en barres de l'importance des variables d'un modèle.

    Cette fonction extrait l'attribut `feature_importances_` d'un modèle entraîné
    (type Random Forest, Decision Tree, XGBoost, etc.), trie les variables par ordre
    décroissant d'importance et génère un diagramme à barres horizontal.

    Paramètres
    ----------
    model : estimator
        Modèle de Machine Learning entraîné possédant l'attribut `feature_importances_`.
    X_train : pd.DataFrame
        DataFrame d'entraînement contenant les variables explicatives (utilisé pour
        récupérer les noms des colonnes).
    y_train : pd.Series ou pd.DataFrame ou list
        Variable cible d'entraînement (utilisée pour afficher le nom de la cible dans le titre).

    Retours
    -------
    pd.DataFrame
        Un DataFrame contenant deux colonnes ('Feature' et 'Importance') trié
        par importance décroissante.
    """
    # 1. Extraction des importances et des noms de variables
    importances = model.feature_importances_
    feature_names = list(X_train.columns)

    # 2. Création et tri du DataFrame
    features_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    # 3. Récupération du nom de la cible pour le titre
    if isinstance(y_train, pd.Series) and y_train.name:
        label_name = y_train.name
    elif isinstance(y_train, pd.DataFrame):
        label_name = ", ".join(y_train.columns)
    else:
        label_name = "Cible"

    # 4. Génération du graphique
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=features_importance_df,
        hue="Feature",
        legend=False,
    )
    plt.title(f"Importance des caractéristiques pour la variable '{label_name}'")
    plt.xlabel("Importance")
    plt.ylabel("Caractéristique")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    return features_importance_df


def shape_modele_status(mod, X_test: pd.DataFrame) -> None:
    """Affiche un graphique Beeswarm (Summary Plot) des valeurs SHAP pour évaluer l'impact des variables.

    Paramètres
    ----------
    mod : BaseEstimator
        Modèle de Machine Learning entraîné basé sur des arbres de décision
        (ex. `RandomForestClassifier`, `XGBClassifier`, `ExtraTreesClassifier`).
    X_test : pd.DataFrame
        DataFrame de test contenant les variables explicatives utilisées pour
        le calcul des contributions SHAP.

    Retours
    -------
    None
        Affiche directement le graphique SHAP Beeswarm dans l'environnement d'exécution.
    """
    if shap is None:
        raise ImportError(
            "Le package 'shap' n'est pas installé. Installe-le avec : pip install shap"
        )

    explainer = shap.TreeExplainer(mod)
    explanation = explainer(X_test)
    if len(explanation.shape) == 3:
        explanation_class1 = explanation[:, :, 1]
    else:
        explanation_class1 = explanation

    shap.plots.beeswarm(explanation_class1)

