"""
Fonctions d'analyse : corrélations entre variables et variables cibles.

Fonctions pures (calcul uniquement, pas d'affichage). Les graphiques associés
(heatmaps, barplots) restent dans `notebooks/exploration.ipynb`, qui peut
appeler ces fonctions puis tracer lui-même les résultats.
"""
import numpy as np
import pandas as pd
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
