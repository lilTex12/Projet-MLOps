"""
Fonctions de nettoyage du jeu de données.

Fonctions pures : elles prennent un DataFrame (+ paramètres) et retournent
un DataFrame, sans jamais lire ou écrire de fichier. C'est ce qui les rend
faciles à tester unitairement (cf. tests/unit_tests/domain/test_nettoyage.py).
"""
import pandas as pd


def identification_valeurs_null(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le nombre et le pourcentage de valeurs nulles pour chaque variable.

    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame à analyser.

    Retours
    -------
    pd.DataFrame
        Un Df contenant pour chaque variable, le nombre et le pourcentage de valeurs null
    """
    nb_nulls = df.isnull().sum()
    pct_nulls = (nb_nulls / len(df)) * 100

    df_null = pd.DataFrame({
        'nb_nulls': nb_nulls,
        'pourcentage_%': pct_nulls.round(2)
    })

    df_null = df_null[df_null['nb_nulls'] > 0].sort_values(
        by='nb_nulls',
        ascending=False
    )

    return df_null


def supression_individus_var_null(df: pd.DataFrame, var: list) -> pd.DataFrame:
    """
    Supprime les individus ayant des valeurs nulles dans les variables spécifiées.

    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame à modifier.
    var : list
        La liste des variables à considérer.

    Retours
    -------
    pd.DataFrame
        Le DataFrame sans les individus ayant des valeurs nulles dans les variables spécifiées.
    """
    return df.dropna(subset=var)


def verifier_bornes_variables_quanti(df: pd.DataFrame, bornes: dict) -> pd.DataFrame:
    """
    Vérifie si les valeurs des variables sont comprises entre leur min et leur max théoriques.

    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame à vérifier.
    bornes : dict
        Dictionnaire au format {'var': (min, max)}.

    Retours
    -------
    pd.DataFrame
        Un df contenant les informations sur les bornes des variables quanti spécifiées dans le dictionnaire
    """
    df_bornes = []

    for col, (val_min, val_max) in bornes.items():
        if col in df.columns:
            hors_bornes = (df[col] < val_min) | (df[col] > val_max)
            nb_hors_bornes = hors_bornes.sum()

            df_bornes.append({
                'variable': col,
                'borne_min': val_min,
                'borne_max': val_max,
                'min_trouve': df[col].min(),
                'max_trouve': df[col].max(),
                'nb_anomalies': nb_hors_bornes,
                'pourcentage_anomalies_%': round((nb_hors_bornes / len(df)) * 100, 2)
            })

    return pd.DataFrame(df_bornes)


def suppr_cols(df: pd.DataFrame, vars: list) -> pd.DataFrame:
    """
    Supprimer les colonnes présentes dans la liste des variables spécifiées.

    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame à modifier.
    vars : list
        La liste des variables à supprimer.

    Retours
    -------
    pd.DataFrame
        Un df sans les colonnes spécifiées.
    """
    return df.drop(columns=vars, errors='ignore')


def encodage_ordinal(df: pd.DataFrame, ordres: dict) -> pd.DataFrame:
    """
    Encodage ordinal pour les variables dont l'ordre a une importance.

    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame à modifier.
    ordres : dict
        Dictionnaire contenant l'ordre de l'encodage des variables
        exemple : {'colonne1': {'Peu': 0, 'Moyen': 1, 'Beaucoup': 2}, ...}

    Retours
    -------
    pd.DataFrame
        Le DataFrame avec les variables encodées sous forme d'entiers ordonnés.
    """
    df_final = df.copy()
    for col, map_dict in ordres.items():
        if col in df_final.columns:
            df_final[col] = df_final[col].map(map_dict)
    return df_final


def encodage_classique(df: pd.DataFrame, vars: list) -> pd.DataFrame:
    """
    Encodage One-Hot pour les variables catégorielles sans notion d'ordre.

    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame à modifier.
    vars : list
        La liste des variables à encoder.

    Retours
    -------
    pd.DataFrame
        Le DataFrame avec les variables transformées.
    """
    return pd.get_dummies(df, columns=vars, drop_first=True, dtype=int)


def verifier_modalites_rares(
    df: pd.DataFrame,
    variables: list[str] | None = None,
    seuil_pourcentage: float = 5.0,
    seuil_effectif: int | None = None,
) -> pd.DataFrame:
    """
    Identifie les modalités rares dans les variables qualitatives.

    Une modalité est considérée comme rare si son pourcentage est inférieur
    ou égal à seuil_pourcentage ou si son effectif est inférieur ou égal
    à seuil_effectif.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant les données à analyser.
    variables : list[str] ou None, par défaut None
        Liste des variables qualitatives à analyser. Si None, toutes les
        variables de type objet, catégorie ou booléen sont utilisées.
    seuil_pourcentage : float, par défaut 5.0
        Pourcentage maximal en dessous duquel une modalité est considérée comme rare.
    seuil_effectif : int ou None, par défaut None
        Effectif maximal en dessous duquel une modalité est considérée comme rare.

    Retours
    -------
    pd.DataFrame
        DataFrame contenant : variable, modalite, effectif, pourcentage_%, modalite_rare.

    Notes
    -----
    Version "domaine" : la partie affichage graphique (matplotlib/seaborn) a
    été retirée volontairement. Elle appartient à `notebooks/exploration.ipynb`,
    pas à la logique métier testable.
    """
    if variables is None:
        variables = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

    variables_absentes = [v for v in variables if v not in df.columns]
    if variables_absentes:
        raise ValueError(f"Variables absentes du DataFrame : {variables_absentes}")

    if len(variables) == 0:
        raise ValueError("Aucune variable qualitative n'a été trouvée.")

    resultats = []
    for variable in variables:
        effectifs = df[variable].value_counts(dropna=False)
        total = effectifs.sum()

        for modalite, effectif in effectifs.items():
            pourcentage = (effectif / total) * 100
            rare_selon_pourcentage = pourcentage <= seuil_pourcentage
            rare_selon_effectif = (
                seuil_effectif is not None and effectif <= seuil_effectif
            )
            modalite_rare = rare_selon_pourcentage or rare_selon_effectif
            modalite_affichee = "Valeur manquante" if pd.isna(modalite) else modalite

            resultats.append({
                "variable": variable,
                "modalite": modalite_affichee,
                "effectif": effectif,
                "pourcentage_%": round(pourcentage, 2),
                "modalite_rare": modalite_rare
            })

    resultats_df = pd.DataFrame(resultats)
    resultats_df = resultats_df.sort_values(
        by=["variable", "pourcentage_%"], ascending=[True, True]
    ).reset_index(drop=True)

    return resultats_df


def verifier_modalites_rares_quanti(
    df: pd.DataFrame,
    variables: list[str] | None = None,
    seuil_pourcentage: float = 5.0,
    seuil_effectif: int | None = None,
) -> pd.DataFrame:
    """
    Même logique que `verifier_modalites_rares`, mais sans restriction de type
    de colonnes (utile après encodage, quand tout est devenu numérique).
    Partie graphique retirée, cf. note dans `verifier_modalites_rares`.
    """
    if variables is None:
        variables = df.columns.tolist()

    variables_absentes = [v for v in variables if v not in df.columns]
    if variables_absentes:
        raise ValueError(f"Variables absentes du DataFrame : {variables_absentes}")

    if len(variables) == 0:
        raise ValueError("Aucune variable qualitative n'a été trouvée.")

    resultats = []
    for variable in variables:
        effectifs = df[variable].value_counts(dropna=False)
        total = effectifs.sum()

        for modalite, effectif in effectifs.items():
            pourcentage = (effectif / total) * 100
            rare_selon_pourcentage = pourcentage <= seuil_pourcentage
            rare_selon_effectif = (
                seuil_effectif is not None and effectif <= seuil_effectif
            )
            modalite_rare = rare_selon_pourcentage or rare_selon_effectif
            modalite_affichee = "Valeur manquante" if pd.isna(modalite) else modalite

            resultats.append({
                "variable": variable,
                "modalite": modalite_affichee,
                "effectif": effectif,
                "pourcentage_%": round(pourcentage, 2),
                "modalite_rare": modalite_rare
            })

    resultats_df = pd.DataFrame(resultats)
    resultats_df = resultats_df.sort_values(
        by=["variable", "pourcentage_%"], ascending=[True, True]
    ).reset_index(drop=True)

    return resultats_df
