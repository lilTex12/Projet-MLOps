"""
Fonctions d'analyse : corrélations entre variables et variables cibles.

Fonctions pures (calcul uniquement, pas d'affichage). Les graphiques associés
(heatmaps, barplots) restent dans `notebooks/exploration.ipynb`, qui peut
appeler ces fonctions puis tracer lui-même les résultats.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import chi2_contingency


def correlation_variable_pass_status(df: pd.DataFrame) -> pd.Series:
    """
    Calcule les corrélations entre la variable cible pass_status et les autres variables.

    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame contenant les données (déjà encodées numériquement).

    Retours
    -------
    pd.Series
        Les corrélations entre pass_status et les autres variables, triées.
    """
    return df.corrwith(df['pass_status']).sort_values(ascending=False)


def correlation_variable_exam_score(df: pd.DataFrame) -> pd.Series:
    """
    Calcule les corrélations entre la variable cible exam_score et les autres variables.

    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame contenant les données (déjà encodées numériquement).

    Retours
    -------
    pd.Series
        Les corrélations entre exam_score et les autres variables, triées.
    """
    return df.corrwith(df['exam_score']).sort_values(ascending=False)


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Calcule le V de Cramér entre deux variables qualitatives."""
    donnees = pd.DataFrame({"x": x, "y": y}).dropna()

    if donnees["x"].nunique() < 2 or donnees["y"].nunique() < 2:
        return np.nan

    tableau_contingence = pd.crosstab(donnees["x"], donnees["y"])
    chi2 = chi2_contingency(tableau_contingence)[0]
    n = tableau_contingence.to_numpy().sum()
    phi2 = chi2 / n

    nombre_lignes, nombre_colonnes = tableau_contingence.shape
    phi2_corr = max(0, phi2 - ((nombre_colonnes - 1) * (nombre_lignes - 1)) / (n - 1))
    nombre_lignes_corr = nombre_lignes - ((nombre_lignes - 1) ** 2) / (n - 1)
    nombre_colonnes_corr = nombre_colonnes - ((nombre_colonnes - 1) ** 2) / (n - 1)
    denominateur = min(nombre_colonnes_corr - 1, nombre_lignes_corr - 1)

    if denominateur <= 0:
        return np.nan

    return np.sqrt(phi2_corr / denominateur)


def _ratio_correlation_eta(variable_numerique: pd.Series, variable_qualitative: pd.Series) -> float:
    """Calcule le ratio de corrélation Eta entre une variable numérique et une qualitative."""
    donnees = pd.DataFrame({
        "numerique": variable_numerique,
        "qualitative": variable_qualitative
    }).dropna()

    if donnees["numerique"].nunique() < 2 or donnees["qualitative"].nunique() < 2:
        return np.nan

    moyenne_generale = donnees["numerique"].mean()
    sommes_carres_inter_groupes = 0

    for _, groupe in donnees.groupby("qualitative", observed=False):
        nombre_observations = len(groupe)
        moyenne_groupe = groupe["numerique"].mean()
        sommes_carres_inter_groupes += nombre_observations * (moyenne_groupe - moyenne_generale) ** 2

    sommes_carres_totales = ((donnees["numerique"] - moyenne_generale) ** 2).sum()

    if sommes_carres_totales == 0:
        return 0.0

    return np.sqrt(sommes_carres_inter_groupes / sommes_carres_totales)


def analyser_correlations_mixtes(
    df: pd.DataFrame,
    variables: list[str] | None = None,
    exclure: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """
    Calcule les associations entre variables numériques et qualitatives.

    Trois méthodes sont utilisées :
    - Pearson pour les relations numérique-numérique ;
    - V de Cramér pour les relations qualitative-qualitative ;
    - Ratio de corrélation Eta pour les relations numérique-qualitative.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant les données à analyser.
    variables : list[str] ou None, par défaut None
        Liste des variables à analyser. Si None, toutes les colonnes du DataFrame.
    exclure : list[str] ou None, par défaut None
        Variables à exclure ("student_id" est exclu automatiquement si présent).

    Retours
    -------
    dict[str, pd.DataFrame]
        "numerique_numerique", "qualitative_qualitative", "numerique_qualitative".
    """
    if exclure is None:
        exclure = []
    if "student_id" in df.columns:
        exclure.append("student_id")

    if variables is None:
        variables_selectionnees = [c for c in df.columns if c not in exclure]
    else:
        variables_absentes = [c for c in variables if c not in df.columns]
        if variables_absentes:
            raise ValueError(f"Variables absentes du DataFrame : {variables_absentes}")
        variables_selectionnees = [c for c in variables if c not in exclure]

    if len(variables_selectionnees) == 0:
        raise ValueError("Aucune variable disponible pour l'analyse.")

    variables_numeriques = df[variables_selectionnees].select_dtypes(include="number").columns.tolist()
    variables_qualitatives = [c for c in variables_selectionnees if c not in variables_numeriques]

    # 1. Numérique-numérique : Pearson
    if len(variables_numeriques) >= 2:
        matrice_num_num = df[variables_numeriques].corr(method="pearson")
    else:
        matrice_num_num = pd.DataFrame(index=variables_numeriques, columns=variables_numeriques)

    # 2. Qualitative-qualitative : V de Cramér
    if len(variables_qualitatives) >= 1:
        matrice_qual_qual = pd.DataFrame(
            index=variables_qualitatives, columns=variables_qualitatives, dtype=float
        )
        for v1 in variables_qualitatives:
            for v2 in variables_qualitatives:
                matrice_qual_qual.loc[v1, v2] = _cramers_v(df[v1], df[v2])
    else:
        matrice_qual_qual = pd.DataFrame()

    # 3. Numérique-qualitative : Eta
    if len(variables_numeriques) >= 1 and len(variables_qualitatives) >= 1:
        matrice_num_qual = pd.DataFrame(
            index=variables_numeriques, columns=variables_qualitatives, dtype=float
        )
        for vn in variables_numeriques:
            for vq in variables_qualitatives:
                matrice_num_qual.loc[vn, vq] = _ratio_correlation_eta(df[vn], df[vq])
    else:
        matrice_num_qual = pd.DataFrame()

    return {
        "numerique_numerique": matrice_num_num,
        "qualitative_qualitative": matrice_qual_qual,
        "numerique_qualitative": matrice_num_qual
    }

def graphique_score_selon_mesure(df: pd.DataFrame, mesure: str | None = None, variable_cible: str = "exam_score", nb_classes: int = 5, decouper_quantitatives: bool = False
) -> None:
    """
    Représente graphiquement la variable cible quantitative (ex: examen_score)
    en fonction d'une ou plusieurs variables du DataFrame.

    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame contenant les données.
    mesure : str ou None, facultatif
        Nom de la variable explicative. Si None, propose un choix interactif.
        Si "toutes", génère tous les graphiques.
    variable_cible : str, par défaut "examen_score"
        Nom de la variable quantitative à analyser.
    nb_classes : int, par défaut 5
        Nombre d'intervalles si decouper_quantitatives=True.
    decouper_quantitatives : bool, par défaut False
        Si True, découpe les variables quantitatives en tranches (boxplot).
        Si False, affiche un nuage de points avec régression linéaire.
    """

    # Vérification de la présence de la variable cible
    if variable_cible not in df.columns:
        raise ValueError(
            f"La variable cible '{variable_cible}' n'existe pas dans le DataFrame."
        )

    # Suppression des lignes où la variable cible est absente
    donnees = df.dropna(subset=[variable_cible]).copy()

    # Liste des variables explicatives disponibles
    variables_exclues = {variable_cible, "student_id"}
    variables_disponibles = [
        col for col in df.columns if col not in variables_exclues
    ]

    if not variables_disponibles:
        raise ValueError("Aucune variable disponible pour l'analyse.")

    # Choix interactif de la variable
    if mesure is None:
        print("Variables disponibles :\n")
        for numero, colonne in enumerate(variables_disponibles, start=1):
            print(f"{numero}. {colonne}")

        choix = input(
            "\nEntrez le numéro de la variable à analyser ou 'toutes' : "
        )

        if choix.lower() == "toutes":
            mesure = "toutes"
        else:
            try:
                numero = int(choix)
                if numero < 1 or numero > len(variables_disponibles):
                    raise ValueError
                mesure = variables_disponibles[numero - 1]
            except ValueError:
                raise ValueError(
                    "Choix invalide. Entrez un numéro valide ou 'toutes'."
                )

    # Vérification de la variable demandée
    if mesure != "toutes" and mesure not in variables_disponibles:
        raise ValueError(
            f"La variable '{mesure}' n'existe pas ou ne peut pas être analysée."
        )

    # Variables à traiter
    a_traiter = variables_disponibles if mesure == "toutes" else [mesure]

    sns.set_theme(style="whitegrid")

    for col in a_traiter:
        plt.figure(figsize=(8, 5))

        # Cas 1 : Variable explicative quantitative
        if pd.api.types.is_numeric_dtype(donnees[col]):
            if decouper_quantitatives:
                # Groupement par tranches avec pd.cut
                col_binned = f"{col}_tranches"
                donnees[col_binned] = pd.cut(donnees[col], bins=nb_classes)
                sns.boxplot(
                    data=donnees,
                    x=col_binned,
                    y=variable_cible,
                    color="#4c72b0"
                )
                plt.xticks(rotation=30)
                plt.xlabel(f"{col} (par tranches)")
            else:
                # Scatterplot + droite de régression
                sns.regplot(
                    data=donnees,
                    x=col,
                    y=variable_cible,
                    scatter_kws={"alpha": 0.6},
                    line_kws={"color": "red"}
                )
                plt.xlabel(col)

            plt.title(f"{variable_cible} en fonction de {col}")

        # Cas 2 : Variable explicative qualitative
        else:
            sns.boxplot(
                data=donnees,
                x=col,
                y=variable_cible,
                color="#55a868"
            )
            plt.xticks(rotation=45 if donnees[col].nunique() > 5 else 0)
            plt.title(f"Distribution de {variable_cible} selon {col}")
            plt.xlabel(col)

        plt.ylabel(variable_cible)
        plt.tight_layout()
        plt.show()

def graphique_pass_selon_mesure(df: pd.DataFrame, mesure: str | None = None, variable_cible: str = "pass_status", nb_classes: int = 5
) -> None:
    """
    Représente graphiquement le nombre d'étudiants ayant réussi ou échoué
    selon une variable choisie.

    Pour une variable qualitative, la fonction affiche le nombre de réussites
    et d'échecs pour chaque catégorie.

    Pour une variable quantitative, les valeurs sont regroupées en classes
    afin de représenter le nombre de réussites et d'échecs par intervalle.

    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame contenant les données des étudiants.

    mesure : str ou None, facultatif
        Nom de la variable à analyser.

        - Si une variable est indiquée, un seul graphique est produit.
        - Si mesure=None, l'utilisateur choisit une variable dans la liste.
        - Si mesure="toutes", un graphique est produit pour toutes les variables
          compatibles.

    variable_cible : str, par défaut "pass_status"
        Nom de la variable indiquant la réussite ou l'échec.

    nb_classes : int, par défaut 5
        Nombre de classes utilisées pour regrouper les variables quantitatives.

    Retours
    -------
    None
        La fonction affiche un ou plusieurs graphiques.
    """

    # Vérification de la présence de la variable cible
    if variable_cible not in df.columns:
        raise ValueError(
            f"La variable cible '{variable_cible}' n'existe pas dans le DataFrame."
        )

    # Suppression des lignes où la variable cible est absente
    donnees = df.dropna(subset=[variable_cible]).copy()

    # Liste des variables disponibles
    variables_exclues = {
        variable_cible,
        "student_id"
    }

    variables_disponibles = [
        colonne for colonne in df.columns
        if colonne not in variables_exclues
    ]

    if len(variables_disponibles) == 0:
        raise ValueError("Aucune variable disponible pour l'analyse.")

    # Choix de la variable par l'utilisateur
    if mesure is None:
        print("Variables disponibles :\n")

        for numero, colonne in enumerate(variables_disponibles, start=1):
            print(f"{numero}. {colonne}")

        choix = input(
            "\nEntrez le numéro de la variable à analyser "
            "ou 'toutes' : "
        )

        if choix.lower() == "toutes":
            mesure = "toutes"
        else:
            try:
                numero = int(choix)

                if numero < 1 or numero > len(variables_disponibles):
                    raise ValueError

                mesure = variables_disponibles[numero - 1]

            except ValueError:
                raise ValueError(
                    "Choix invalide. Entrez un numéro valide ou 'toutes'."
                )

    # Vérification du nom de la variable
    if mesure != "toutes" and mesure not in variables_disponibles:
        raise ValueError(
            f"La variable '{mesure}' n'existe pas ou ne peut pas être analysée."
        )

    # Configuration graphique
    sns.set_theme(style="whitegrid")
