"""
Fonctions liées à la préparation des données pour la modélisation.

C'est ici que tu ajouteras aussi tes futures fonctions d'entraînement de
modèle "pures" (ex: `entrainer_modele(X_train, y_train, params) -> model`),
que `application/train.py` viendra orchestrer avec MLflow.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def separer_train_test(
    df: pd.DataFrame,
    variable_cible: str,
    proportion_test: float = 0.2,
    random_state: int = 42,
    stratifier: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Sépare un DataFrame en jeux de données d'entraînement et de test.

    La variable cible est séparée des variables explicatives. Une stratification
    peut être utilisée afin de conserver une répartition similaire de la cible
    dans les jeux d'entraînement et de test.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant les variables explicatives et la variable cible.
    variable_cible : str
        Nom de la colonne à prédire.
    proportion_test : float, par défaut 0.2
        Proportion des observations utilisées pour le jeu de test.
    random_state : int, par défaut 42
        Graine utilisée pour obtenir une séparation reproductible.
    stratifier : bool, par défaut True
        Si True, conserve approximativement la même répartition des modalités
        de la variable cible dans les jeux d'entraînement et de test.

    Retours
    -------
    tuple
        X_train, X_test, y_train, y_test
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Le paramètre 'df' doit être un DataFrame pandas.")

    if variable_cible not in df.columns:
        raise ValueError(f"La variable cible '{variable_cible}' n'existe pas dans le DataFrame.")

    if not 0 < proportion_test < 1:
        raise ValueError("La proportion_test doit être comprise entre 0 et 1.")

    donnees = df.dropna(subset=[variable_cible]).copy()
    X = donnees.drop(columns=[variable_cible])
    y = donnees[variable_cible]

    valeur_stratification = y if stratifier else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=proportion_test,
        random_state=random_state,
        stratify=valeur_stratification
    )

    return X_train, X_test, y_train, y_test
