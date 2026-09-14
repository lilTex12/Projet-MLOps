"""
Accès aux données brutes : téléchargement Kaggle et lecture des fichiers.

Toute fonction qui touche à une ressource externe (fichier, API, DB) vit ici,
et nulle part ailleurs. Le reste du code (domain, application) ne doit jamais
appeler kagglehub ou pd.read_csv directement.
"""
import glob
import os

import kagglehub
import pandas as pd


def telecharger_dataset_kaggle(dataset_id: str = "mobeenfatimah/student-exam-performance-and-success-dataset") -> str:
    """
    Télécharge le dataset Kaggle et retourne le chemin du dossier local.

    Paramètres
    ----------
    dataset_id : str
        Identifiant du dataset sur Kaggle (format 'owner/dataset-name').

    Retours
    -------
    str
        Chemin du dossier contenant les fichiers téléchargés.
    """
    dossier = kagglehub.dataset_download(dataset_id)
    print("Dataset téléchargé dans :", dossier)
    return dossier


def lister_fichiers_csv(dossier: str) -> list[str]:
    """
    Liste tous les fichiers CSV présents dans un dossier (récursif).

    Paramètres
    ----------
    dossier : str
        Dossier dans lequel chercher.

    Retours
    -------
    list[str]
        Liste des chemins de fichiers CSV trouvés.
    """
    fichiers = glob.glob(os.path.join(dossier, "**", "*"), recursive=True)
    return [f for f in fichiers if f.endswith(".csv")]


def charger_csv(chemin_fichier: str) -> pd.DataFrame:
    """
    Charge un fichier CSV en DataFrame.

    Paramètres
    ----------
    chemin_fichier : str
        Chemin vers le fichier CSV.

    Retours
    -------
    pd.DataFrame
        Les données chargées.
    """
    return pd.read_csv(chemin_fichier)


def charger_dataset_local(chemin_fichier: str = "data/raw/student_exam_performance.csv") -> pd.DataFrame:
    """
    Charge directement le CSV déjà présent dans le repo (data/raw/), sans
    repasser par un téléchargement Kaggle. À utiliser une fois le dataset
    versionné dans le repo (cf. data/dataset-metadata.json).
    """
    return pd.read_csv(chemin_fichier)
