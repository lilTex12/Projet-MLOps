from student_perf.infrastructure.extract import (
    telecharger_dataset_kaggle,
    lister_fichiers_csv,
    charger_csv,
    charger_dataset_local,
)

import pandas as pd 
import numpy as np



def recuperer_dataset_kaggle(
    dataset_id: str = "mobeenfatimah/student-exam-performance-and-success-dataset",
    index_fichier: int = 0,
    afficher_fichiers: bool = True,
) -> tuple[pd.DataFrame, str, list[str]]:
  """
    Télécharge le dataset Kaggle, liste les fichiers CSV trouvés et charge le DataFrame.

    Paramètres
    ----------
    dataset_id : str
        Identifiant du dataset sur Kaggle.
    index_fichier : int, par défaut 0
        Index du fichier CSV à charger dans la liste des CSV détectés.
    afficher_fichiers : bool, par défaut True
        Affiche la liste des fichiers CSV trouvés.

    Retours
    -------
    tuple[pd.DataFrame, str, list[str]]
        (df, dossier, fichiers_csv)
    """
  dossier = telecharger_dataset_kaggle(dataset_id)
  fichiers_csv = lister_fichiers_csv(dossier)

  if afficher_fichiers:
    print("Fichiers CSV trouvés :", fichiers_csv)

  if not fichiers_csv:
    raise FileNotFoundError(
        f"Aucun fichier CSV trouvé dans le dossier : {dossier}"
    )

  if index_fichier < 0 or index_fichier >= len(fichiers_csv):
    raise IndexError(
        f"index_fichier={index_fichier} hors limites ({len(fichiers_csv)}"
        " fichier(s) trouvé(s))."
    )

  df = charger_csv(fichiers_csv[index_fichier])
  return df, dossier, fichiers_csv


__all__ = [
    "telecharger_dataset_kaggle",
    "lister_fichiers_csv",
    "charger_csv",
    "charger_dataset_local",
    "recuperer_dataset_kaggle",
]