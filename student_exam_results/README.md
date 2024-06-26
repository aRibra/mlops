# MLOPs - Student Exam Results
Toy example for predicting Student Exam Results


### Intro

The `student_exam_result.py` contains all the definitions and classes for the _Student Exam Result_ project.

The notebook `Student_Exam_Result.ipynb` is used to demonstrate the usage of the code.

Below, we use the same dataset to build multiple models (_LogisticRegression_) each responsible for predicting categorical encoded values of:
- `MatchScore`
- `WritingScore`
- `ReadingScore`


### `StudentExamResultFramework` class
This class is the main class for the framework. It has a main method `run_pipeline()` which runs the whole system pipeline including:
- Analyzing data
- Cleaning data
- Splitting dataset to train and test sets
- Training the model
- Evaluating the model on the test data
- Run Explainable AI module on the output model using `shapely`
- Encrypts/Decrypts the dataset

#### Constructor
The constructor takes the student exam result CSV dataset path (`csv_file_path`), and number of bins for the target score we want to predict (`nb_bins`).

#### `run_pipeline()`
This is the main method in the class, and it takes one argument (`target_column`) which is the target column in the dataset that we want to predict. The possible values are:
- "MathScore"
- "WritingScore"
- "ReadingScore"

When selecting one of these target columns, the column will be encoded categorically, and the model will be trained to predict the categories. The number of categories is set by `nb_bins`. By default, the number of bins (classes) is `5` which represents the following categories:
- 0: (0 - 20]
- 1: (20 - 40]
- 2: (40 - 60]
- 3: (60 - 80]
- 4: (80 - 100]

You can choose a higher number of bins which will have more bins with smaller score intervals. Or, choose a lower number of bins which will have fewer bins with larger score intervals.

### Dataset
- students-exam-scores
- Kaggle link: https://www.kaggle.com/datasets/desalegngeb/students-exam-scores

### CI/CD
Provided a yaml file `.github/workflows/app.yml` configuring a build run before authorizing a commit. The workflow runs tests configured in `tests/test_student_framework.py` using pytest.
