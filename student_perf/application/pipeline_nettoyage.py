"""
Ce module appelle uniquement les fonctions de `domain/` et `infrastructure/`, il permet d'obtenir le jeu de données final

Usage :
    python -m student_perf.application.pipeline_nettoyage
"""
import pandas as pd

from student_perf.domain.nettoyage import (
    identification_valeurs_null,
    supression_individus_var_null,
    verifier_bornes_variables_quanti,
    suppr_cols,
    encodage_ordinal,
    encodage_classique,
)
from student_perf.settings.base import (
    REGLES_BORNES_THEORIQUES,
    DICT_VAR_QUALI_ORDINAL,
)


def nettoyer_donnees_brutes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Étape 1 : suppression des colonnes inutiles + vérification des bornes.
    """
    df = suppr_cols(df, ['student_id'])

    df_bornes = verifier_bornes_variables_quanti(df, REGLES_BORNES_THEORIQUES)
    if (df_bornes['nb_anomalies'] > 0).any():
        print("⚠️ Anomalies de bornes détectées :")
        print(df_bornes[df_bornes['nb_anomalies'] > 0])

    return df


def preparer_jeu_donnees(
    df: pd.DataFrame,
    variables_reinjectees: list[str],
    variables_a_supprimer: list[str],
) -> pd.DataFrame:
    df_prepare = supression_individus_var_null(df, variables_reinjectees)
    df_prepare = suppr_cols(df_prepare, variables_a_supprimer)

    variables_quali = df_prepare.select_dtypes(include=['object', 'category']).columns.tolist()
    variables_quali_non_ordinales = [
        v for v in variables_quali if v not in DICT_VAR_QUALI_ORDINAL
    ]

    df_prepare = encodage_classique(df_prepare, variables_quali_non_ordinales)
    df_prepare = encodage_ordinal(df_prepare, DICT_VAR_QUALI_ORDINAL)

    return df_prepare


def run_pipeline(df_brut: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = nettoyer_donnees_brutes(df_brut)

    # Colonnes réinjectées / supprimées identifiées dans le notebook
    # (cellules 49, 51, 54, 55) : ces variables ne sont pas significatives
    # pour prédire la cible, donc on récupère les individus qui les avaient nulles.
    variables_reinjectees = ['attendance_percentage', 'time_management_score', 'previous_gpa']
    variables_a_supprimer = ['notes_quality', 'sleep_quality', 'parent_education', 'device_availability']

    df_status = preparer_jeu_donnees(df, variables_reinjectees, variables_a_supprimer)
    df_score = preparer_jeu_donnees(df, variables_reinjectees, variables_a_supprimer)

    return {"df_status": df_status, "df_score": df_score}


if __name__ == "__main__":
    from student_perf.infrastructure.extract import charger_dataset_local

    df_brut = charger_dataset_local()
    resultats = run_pipeline(df_brut)

    resultats["df_score"].to_csv("data/jeu_donnees_final.csv", index=False)
    print("Pipeline terminé. Jeu de données final exporté dans data/jeu_donnees_final.csv")
