Dataset de Performance aux Examens et de Réussite Académique des Étudiants

Ce dataset présente croise les habitudes d'études, les modes de vies, le bien-être et les facteurs académiques pour prédire la réussite aux examens de 100 000 étudiants.

❓Quelques informations : 
- 100 000 étudiants uniques (aucun doublon)
- 44 variables
- Taux de réussite global : ~77,4 %
Distribution des notes (courbe académique normale) :
D : ~34,8 %
C : ~21,8 %
F : ~22,6 %
B : ~13,2 %
A : ~7,5 %
- dernier update : 23 days ago

💬Mots clés : éducation, étudiants, performance académique, examens, réussite, bien-être, stress, notes ... 

Plusieurs variables cibles possibles pour des modèles de régression ou de classification : 
- exam_score (Numérique / Continue) : Score final allant de 0 à 100. Idéal pour la régression.
- pass_status (Binaire / Catégorielle) : Pass (≥ 50) ou Fail (< 50). Idéal pour la classification binaire.
- performance_grade (Catégorielle multi-classes) : Notes lettrées standards (A, B, C, D, F) basées sur une distribution académique réaliste (~77 % de réussite).
- performance_level (Catégorielle multi-classes) : Catégories de performance (High, Medium, Low).

Détails des variables 

44 variables réparties en 6 thèmes :

🎓1. Démographie des étudiants

student_id, age, gender, education_level, school_type, family_income, parent_education, urban_rural

💯2. Base académique

previous_exam_score, previous_gpa, attendance_percentage, assignment_completion_rate, class_participation, private_tuition

📖3. Habitudes et comportements d'étude

study_hours_per_day, self_study_hours, study_consistency, study_environment, study_method, revision_frequency, practice_tests_completed, notes_quality

😴4. Mode de vie et bien-être

sleep_hours, sleep_quality, daily_screen_time, physical_activity_hours, break_frequency, stress_level, motivation_level

💻5. Technologie et facteurs liés à l'examen

internet_access, device_availability, educational_app_usage, online_course_hours, exam_difficulty, exam_preparation_days, questions_attempted, questions_correct, time_management_score, exam_anxiety_level

🎯6. Cibles construites (engineered targets)

exam_score, performance_grade, pass_status, performance_level


Lien vers le dataset : [https://www.kaggle.com/datasets/mobeenfatimah/student-exam-performance-and-success-dataset?resource=download](https://www.kaggle.com/datasets/mobeenfatimah/student-exam-performance-and-success-dataset)
Tout droits réservés @mobeenfatimah / Mobeen Fatima

