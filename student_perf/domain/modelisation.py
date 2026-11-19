"""
Fonctions liées à la préparation des données pour la modélisation.

C'est ici que tu ajouteras aussi tes futures fonctions d'entraînement de
modèle "pures" (ex: `entrainer_modele(X_train, y_train, params) -> model`),
que `application/train.py` viendra orchestrer avec MLflow.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    
    
def separer_train_test_regressor(
    df: pd.DataFrame,
    variable_cible: str,
    proportion_test: float = 0.2,
    random_state: int = 42,
    stratifier: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Sépare un DataFrame en jeux de données d'entraînement et de test pour un modèle de régression ou de classification.


    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame contenant les variables explicatives et la variable cible.
    variable_cible : str
        Le nom de la colonne à prédire.
    proportion_test : float, default=0.2
        La proportion des observations allouées au jeu de test (ex. 0.2 pour 20%).
    random_state : int, default=42
        La graine aléatoire pour garantir la reproductibilité du découpage.
    stratifier : bool, default=True
        Si True, tente de préserver les proportions de la variable cible dans les
        ensembles d'entraînement et de test.

    Retours
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        Un tuple contenant dans cet ordre :
        - X_train : Variables explicatives du jeu d'entraînement.
        - X_test : Variables explicatives du jeu de test.
        - y_train : Variable cible du jeu d'entraînement.
        - y_test : Variable cible du jeu de test.

    Exceptions
    ----------
    TypeError
        Si `df` n'est pas un DataFrame Pandas.
    ValueError
        Si `variable_cible` n'est pas dans les colonnes du DataFrame, ou si
        `proportion_test` n'est pas comprise strictement entre 0 et 1.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Le paramètre 'df' doit être un DataFrame pandas.")

    if variable_cible not in df.columns:
        raise ValueError(
            f"La variable cible '{variable_cible}' n'existe pas dans le DataFrame."
        )

    if not 0 < proportion_test < 1:
        raise ValueError("La proportion_test doit être comprise entre 0 et 1.")

    donnees = df.dropna(subset=[variable_cible]).copy()

    X = donnees.drop(columns=[variable_cible])
    y = donnees[variable_cible]

    # Vérification automatique si la stratification est possible
    if stratifier:
        counts = y.value_counts()
        if (counts < 2).any():
            print(
                "Avertissement : Certaines classes apparaissent moins de 2 fois. "
                "Désactivation automatique de la stratification."
            )
            stratifier = False

    valeur_stratification = y if stratifier else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=proportion_test,
        random_state=random_state,
        stratify=valeur_stratification,
    )

    return X_train, X_test, y_train, y_test
    
def random_forest_classifier(
    X_train: pd.DataFrame,
    y_train: list,
    X_test: pd.DataFrame,
    y_test: list,
    n_estimators_list: list,
    max_depth_list: list,
    cv: int = 5,
) -> tuple[RandomForestClassifier, dict, np.ndarray]:
    """Optimise les hyperparamètres d'un RandomForestClassifier par validation croisée et évalue le modèle sur un jeu de test.

    Paramètres
    ----------
    X_train : pd.DataFrame
        Le DataFrame contenant les variables explicatives d'entraînement.
    y_train : list
        La liste contenant la variable cible d'entraînement.
    X_test : pd.DataFrame
        Le DataFrame contenant les variables explicatives de test.
    y_test : list
        La liste contenant la variable cible de test.
    n_estimators_list : list[int]
        La liste des nombres d'arbres à tester.
    max_depth_list : list[int]
        La liste des profondeurs maximales d'arbre à tester.
    cv : int, default=5
        Le nombre de plis pour la validation croisée.

    Retours
    -------
    tuple[RandomForestClassifier, dict, np.ndarray]
        Un tuple contenant le meilleur modèle entraîné, un dictionnaire avec
        ses meilleurs paramètres et un tableau NumPy avec les prédictions sur
        le jeu de test.
    """

    # 1. Définition de la grille d'hyperparamètres
    param_grid = {
        "n_estimators": n_estimators_list,
        "max_depth": max_depth_list,
    }

    # 2. Recherche par grille avec Validation Croisée sur X_train
    rf_base = RandomForestClassifier(class_weight='balanced' , random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    # Récupération du meilleur modèle entraîné
    best_mod = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # 3. Graphique des résultats de la validation croisée
    scores_matrix = grid_search.cv_results_["mean_test_score"].reshape(
        len(max_depth_list), len(n_estimators_list)
    )

    plt.figure(figsize=(8, 5))
    for i, depth in enumerate(max_depth_list):
        plt.plot(
            n_estimators_list,
            scores_matrix[i, :],
            label=f"max_depth = {depth}",
        )

    plt.xlabel("n_estimators")
    plt.ylabel("Score Moyen (Accuracy CV)")
    plt.title("Variation du score en fonction de n_estimators et max_depth")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 4. Évaluation finale sur l'ensemble de TEST
    y_pred = best_mod.predict(X_test)

    # 5. Affichage des résultats
    print("--- RÉSULTATS DE L'OPTIMISATION ---")
    print("Meilleurs paramètres trouvés :", best_params)

    return best_mod, best_params, y_pred
    
def random_forest_regressor(
    X_train: pd.DataFrame,
    y_train: list,
    X_test: pd.DataFrame,
    y_test: list,
    n_estimators_list: list,
    max_depth_list: list,
    cv: int = 5,
) -> tuple[RandomForestRegressor, dict, list]:
    """Optimise les hyperparamètres d'un RandomForestRegressor par validation croisée et évalue le modèle sur un jeu de test.

    Paramètres
    ----------
    X_train : pd.DataFrame
        Le DataFrame contenant les variables explicatives d'entraînement.
    y_train : list
        La liste contenant la variable cible d'entraînement.
    X_test : pd.DataFrame
        Le DataFrame contenant les variables explicatives de test.
    y_test : list
        La liste contenant la variable cible de test.
    n_estimators_list : list
        La liste des nombres d'arbres à tester.
    max_depth_list : list
        La liste des profondeurs maximales d'arbre à tester.
    cv : int, default=5
        Le nombre de plis pour la validation croisée.

    Retours
    -------
    tuple[RandomForestRegressor, dict, list]
        Un tuple contenant le meilleur modèle entraîné, un dictionnaire avec
        ses meilleurs paramètres et une liste avec les prédictions sur le jeu
        de test.
    """
    # 1. Définition de la grille d'hyperparamètres
    param_grid = {
        "n_estimators": n_estimators_list,
        "max_depth": max_depth_list,
    }

    # 2. Recherche par grille avec Validation Croisée sur X_train (scoring='r2')
    rf_base = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    # Récupération du meilleur modèle entraîné
    best_mod = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # 3. Graphique des résultats de la validation croisée
    scores_matrix = grid_search.cv_results_["mean_test_score"].reshape(
        len(max_depth_list), len(n_estimators_list)
    )

    plt.figure(figsize=(8, 5))
    for i, depth in enumerate(max_depth_list):
        plt.plot(
            n_estimators_list,
            scores_matrix[i, :],
            label=f"max_depth = {depth}",
        )

    plt.xlabel("n_estimators")
    plt.ylabel("Score Moyen ($R^2$ CV)")
    plt.title("Variation du score en fonction de n_estimators et max_depth")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 4. Évaluation finale sur l'ensemble de TEST
    y_pred = best_mod.predict(X_test)

    # 5. Affichage des résultats
    print("--- RÉSULTATS DE L'OPTIMISATION ---")
    print("Meilleurs paramètres trouvés :", best_params)
    print(f"R² sur le jeu de test : {r2_score(y_test, y_pred):.4f}")
    print(
        f"RMSE sur le jeu de test : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}"
    )

    return best_mod, best_params, y_pred.tolist()
    
def gradient_boosting_regressor(
    X_train: pd.DataFrame,
    y_train: list,
    X_test: pd.DataFrame,
    y_test: list,
    n_estimators_list: list,
    learning_rate_list: list,
    max_depth_list: list,
    cv: int = 5,
) -> tuple[GradientBoostingRegressor, dict, list]:
    """Optimise les hyperparamètres d'un GradientBoostingRegressor par validation croisée et évalue le modèle sur un jeu de test.

    Paramètres
    ----------
    X_train : pd.DataFrame
        Le DataFrame contenant les variables explicatives d'entraînement.
    y_train : list
        La liste contenant la variable cible d'entraînement.
    X_test : pd.DataFrame
        Le DataFrame contenant les variables explicatives de test.
    y_test : list
        La liste contenant la variable cible de test.
    n_estimators_list : list
        La liste des nombres d'arbres à tester.
    learning_rate_list : list
        La liste des taux d'apprentissage à tester.
    max_depth_list : list
        La liste des profondeurs maximales d'arbre à tester.
    cv : int, default=5
        Le nombre de plis pour la validation croisée.

    Retours
    -------
    tuple[GradientBoostingRegressor, dict, list]
        Un tuple contenant le meilleur modèle entraîné, un dictionnaire avec
        ses meilleurs paramètres et une liste avec les prédictions sur le jeu
        de test.
    """
    # 1. Définition de la grille d'hyperparamètres
    param_grid = {
        "n_estimators": n_estimators_list,
        "learning_rate": learning_rate_list,
        "max_depth": max_depth_list,
    }

    # 2. Recherche par grille avec Validation Croisée sur X_train (scoring='r2')
    gb_base = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=gb_base,
        param_grid=param_grid,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    # Récupération du meilleur modèle entraîné
    best_mod = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # 3. Graphique des résultats de la validation croisée
    # Représentation matricielle en (max_depth, learning_rate, n_estimators)
    scores_matrix = grid_search.cv_results_["mean_test_score"].reshape(
        len(max_depth_list), len(learning_rate_list), len(n_estimators_list)
    )

    plt.figure(figsize=(10, 6))
    for i, depth in enumerate(max_depth_list):
        for j, lr in enumerate(learning_rate_list):
            plt.plot(
                n_estimators_list,
                scores_matrix[i, j, :],
                label=f"depth={depth}, lr={lr}",
            )

    plt.xlabel("n_estimators")
    plt.ylabel("Score Moyen ($R^2$ CV)")
    plt.title("Variation du score en fonction des hyperparamètres du Gradient Boosting")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4. Évaluation finale sur l'ensemble de TEST
    y_pred = best_mod.predict(X_test)

    # 5. Affichage des résultats
    print("--- RÉSULTATS DE L'OPTIMISATION ---")
    print("Meilleurs paramètres trouvés :", best_params)
    print(f"R² sur le jeu de test : {r2_score(y_test, y_pred):.4f}")
    print(
        f"RMSE sur le jeu de test : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}"
    )

    return best_mod, best_params, y_pred.tolist()
 
def gradient_boosting_classifier(
    X_train: pd.DataFrame,
    y_train: list,
    X_test: pd.DataFrame,
    y_test: list,
    n_estimators_list: list,
    learning_rate_list: list,
    max_depth_list: list,
    cv: int = 5,
) -> tuple[GradientBoostingClassifier, dict, np.ndarray]:
    """Optimise les hyperparamètres d'un GradientBoostingClassifier par validation croisée et évalue le modèle sur un jeu de test.

    Paramètres
    ----------
    X_train : pd.DataFrame
        Le DataFrame contenant les variables explicatives d'entraînement.
    y_train : list
        La liste contenant la variable cible d'entraînement.
    X_test : pd.DataFrame
        Le DataFrame contenant les variables explicatives de test.
    y_test : list
        La liste contenant la variable cible de test.
    n_estimators_list : list[int]
        La liste des nombres d'arbres à tester.
    learning_rate_list : list[float]
        La liste des taux d'apprentissage à tester.
    max_depth_list : list[int]
        La liste des profondeurs maximales d'arbre à tester.
    cv : int, default=5
        Le nombre de plis pour la validation croisée.

    Retours
    -------
    tuple[GradientBoostingClassifier, dict, np.ndarray]
        Un tuple contenant le meilleur modèle entraîné, un dictionnaire avec
        ses meilleurs paramètres et un tableau NumPy avec les prédictions sur
        le jeu de test.
    """

    # 1. Définition de la grille d'hyperparamètres
    param_grid = {
        "n_estimators": n_estimators_list,
        "learning_rate": learning_rate_list,
        "max_depth": max_depth_list,
    }

    #Equilibre du modèle
    sample_weights = compute_sample_weight("balanced", y_train)

    # 2. Recherche par grille avec Validation Croisée sur X_train
    gb_base = GradientBoostingClassifier( random_state=42)
    grid_search = GridSearchCV(
        estimator=gb_base,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train, sample_weight=sample_weights)

    # Récupération du meilleur modèle entraîné
    best_mod = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # 3. Graphique des résultats de la validation croisée
    scores_matrix = grid_search.cv_results_["mean_test_score"].reshape(
        len(max_depth_list), len(learning_rate_list), len(n_estimators_list)
    )

    plt.figure(figsize=(10, 6))
    for i, depth in enumerate(max_depth_list):
        for j, lr in enumerate(learning_rate_list):
            plt.plot(
                n_estimators_list,
                scores_matrix[i, j, :],
                label=f"depth={depth}, lr={lr}",
            )

    plt.xlabel("n_estimators")
    plt.ylabel("Score Moyen (Accuracy CV)")
    plt.title("Variation de l'Accuracy en fonction des hyperparamètres du Boosting")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4. Évaluation finale sur l'ensemble de TEST
    y_pred = best_mod.predict(X_test)

    # 5. Affichage des résultats
    print("--- RÉSULTATS DE L'OPTIMISATION ---")
    print("Meilleurs paramètres trouvés :", best_params)
    print(f"Accuracy sur le jeu de test : {accuracy_score(y_test, y_pred):.4f}")

    return best_mod, best_params, y_pred
 
