from student_exam_result import StudentExamResultFramework

data_csv_path = 'Students_Exam_Scores/Expanded_data_with_more_features.csv'

ser_driver = StudentExamResultFramework(data_csv_path)

target_column = 'MathScore'

ser_driver.run_pipeline(target_column)

