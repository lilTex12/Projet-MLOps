"""
Fonctions d'évaluation des modèles entraînés.
"""
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def evaluation_modele_classifier(y_test: list, y_pred: list) -> tuple[float, float, float, list]:
    """
    Calcule et affiche les métriques d'évaluation d'un modèle classifier.
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