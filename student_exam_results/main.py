from src.student_exam_result import StudentExamResultFramework


def main():

    data_csv_path = 'data/Expanded_data_with_more_features.csv'

    ser_driver = StudentExamResultFramework(data_csv_path)

    target_column = 'MathScore'

    ser_driver.run_pipeline(target_column)

if __name__ == "__main__":
    main()