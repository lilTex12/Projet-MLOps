"""
Constantes et configuration centralisées.

Avant : ces dictionnaires étaient définis en dur au milieu du notebook
(cellules 36 et 44). Maintenant : un seul endroit à modifier si les règles
métier changent, importé partout où c'est nécessaire.
"""

# Bornes théoriques des variables quantitatives (cellule 36 du notebook)
REGLES_BORNES_THEORIQUES = {
    # note sur 10
    'previous_gpa': (0, 10),
    'stress_level': (0, 10),
    'exam_anxiety_level': (0, 10),

    # Pourcentages
    'time_management_score': (0, 100),
    'attendance_percentage': (0, 100),
    'exam_score': (0, 100),
    'final_grade': (0, 100),
    'previous_examen_score': (0, 100),
    'assignment_completion_rate': (0, 100),
    'questions_attempted': (0, 100),
    'questions_correct': (0, 100),

    # Temps et durées
    'study_hours_per_day': (0, 24),
}

# Variables qualitatives ordinales et leur mapping (cellule 44 du notebook)
DICT_VAR_QUALI_ORDINAL = {
    'family_income': {'Low': 4, 'Lower-Middle': 3, 'Middle': 2, 'Upper-Middle': 1, 'High': 0},
    'parent_education': {'High School': 4, 'Associate': 3, 'Bachelor': 2, 'Master': 1, 'Doctorate': 0},
    'class_participation': {'Low': 2, 'Medium': 1, 'High': 0},
    'study_consistency': {'Low': 2, 'Medium': 1, 'High': 0},
    'study_environment': {'Quiet': 2, 'Moderate': 1, 'Noisy': 0},
    'notes_quality': {'Poor': 2, 'Average': 1, 'Excellent': 0},
    'sleep_quality': {'Poor': 3, 'Fair': 2, 'Good': 1, 'Excellent': 0},
    'break_frequency': {'Rarely': 2, 'Occasionally': 1, 'Frequently': 0},
    'motivation_level': {'Low': 2, 'Medium': 1, 'High': 0},
    'educational_app_usage': {'Low': 2, 'Moderate': 1, 'High': 0},
    'exam_difficulty': {'Easy': 2, 'Medium': 1, 'Hard': 0},
    'performance_grade': {'F': 4, 'D': 3, 'C': 2, 'B': 1, 'A': 0},
    'performance_level': {'Low': 2, 'Medium': 1, 'High': 0},
    'pass_status': {'Fail': 0, 'Pass': 1},
}

CHEMIN_DATA_RAW = "data/raw/student_exam_performance.csv"
