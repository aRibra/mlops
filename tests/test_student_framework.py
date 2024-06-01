import pytest
import pandas as pd
from student_exam_result import StudentExamData, DataPreprocessor, Model, ModelEvaluator


# Dummy data for testing
dummy_data = {
    # 'Unnamed: 0': [1, 2, 3],
    'Gender': ['Male', 'Female', 'Female'],
    'EthnicGroup': ['Group A', 'Group B', 'Group C'],
    'ParentEduc': ['High School', 'Bachelor', 'Master'],
    'LunchType': ['Standard', 'Free/Reduced', 'Standard'],
    'TestPrep': ['None', 'Completed', 'None'],
    'ParentMaritalStatus': ['Married', 'Single', 'Married'],
    'PracticeSport': ['Yes', 'No', 'Yes'],
    'IsFirstChild': ['Yes', 'No', 'Yes'],
    'NrSiblings': [1, 2, 3],
    'TransportMeans': ['Car', 'Bus', 'Bike'],
    'WklyStudyHours': ['<5', '5-10', '10-15'],
    'MathScore': [65, 78, 90],
    'ReadingScore': [70, 80, 85],
    'WritingScore': [75, 82, 88]
}

dummy_df = pd.DataFrame(dummy_data)

def test_student_exam_data():
    student_data = StudentExamData('tests/dummy_data.csv')
    assert not student_data.data.empty, "Data should be loaded correctly"

def test_data_preprocessor():
    preprocessor = DataPreprocessor(dummy_df)
    cleaned_data, label_encoders = preprocessor.clean_data(target_column='MathScore', bins=3)
    target_column = 'MathScore' + 'Cat'
    labels = [str(li) for li in label_encoders[target_column].inverse_transform( list(range(3)) )]
    assert len(labels) == 3, "Labels created from the 'MathScoreCat' label encoder should be exactly 3."
    assert 'MathScoreCat' in cleaned_data.columns, "MathScoreCat column should be present"
    assert not cleaned_data.isnull().values.any(), "There should be no missing values in cleaned data"

def test_model_split():
    preprocessor = DataPreprocessor(dummy_df)
    cleaned_data, label_encoders = preprocessor.clean_data(target_column='MathScore', bins=3)
    model = Model(cleaned_data)
    X_train, X_test, y_train, y_test = model.split_data(target_column='MathScoreCat')
    assert not X_train.empty, "X_train should not be empty"
    assert not X_test.empty, "X_test should not be empty"

def test_build_model():
    preprocessor = DataPreprocessor(dummy_df)
    cleaned_data, label_encoders = preprocessor.clean_data(target_column='MathScore', bins=3)
    model = Model(cleaned_data)
    assert model, "Model object should not be None"

def test_model_evaluator():
    preprocessor = DataPreprocessor(dummy_df)
    cleaned_data, label_encoders = preprocessor.clean_data(target_column='MathScore', bins=3)
    model = Model(cleaned_data)
    X_train, X_test, y_train, y_test = model.split_data(target_column='MathScoreCat')
    trained_model = model.train_model(X_train, y_train)
    evaluator = ModelEvaluator(trained_model, X_test, y_test, labels=['Low', 'Medium', 'High'])
    assert evaluator, "Evaluator object should not be None"
