import pandas as pd
import numpy as np


# Create synthetic dataset
np.random.seed(42)
N = 500
attendance = np.clip(np.random.normal(85, 8, N), 40, 100).round(1)
study_hours = np.clip(np.random.normal(12, 4, N), 0, 40).round(1)
internal_marks = np.clip(np.random.normal(30, 6, N), 0, 40).round(1)
previous_percentage = np.clip(np.random.normal(70, 12, N), 20, 100).round(1)
assignments_completed = np.random.randint(0, 6, N)
class_test_score = np.clip(np.random.normal(18, 4, N), 0, 25).round(1)


# Suppose final_score depends on these with some noise
final_score = (
0.25 * attendance +
1.8 * study_hours +
0.9 * internal_marks +
0.3 * previous_percentage +
2.0 * class_test_score +
1.5 * assignments_completed +
np.random.normal(0, 7, N)
)
final_score = np.clip(final_score / 3.2, 0, 100).round(1) # scale to 0-100


# Create dataframe
df = pd.DataFrame({
'attendance': attendance,
'study_hours': study_hours,
'internal_marks': internal_marks,
'previous_percentage': previous_percentage,
'assignments_completed': assignments_completed,
'class_test_score': class_test_score,
'final_score': final_score
})


# Save
df.to_csv('dataset/students.csv', index=False)
print('Saved dataset/students.csv with', len(df), 'rows')