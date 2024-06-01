import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns
import matplotlib.pyplot as plt
import shap

from cryptography.fernet import Fernet


class StudentExamData:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
    
    def analyze_data(self):
        print("Data Analysis")
        print(self.data.info())
        print("\nStatistics")
        # print(self.data.describe(include='all'))
        return self.data.describe(include='all')

class DataPreprocessor:
    def __init__(self, data):
        self.data = data
        
    def clean_data(self, target_column, bins = 5):
        self.data = self.data.drop(columns=['Unnamed: 0'])
        self.data = self.data.dropna()

        extra_target_cat = None
        
        if target_column == 'MathScore':
            self.data['MathScoreCat'] = pd.cut(self.data['MathScore'], bins=bins, labels=None)
            self.data = self.data.drop(columns=['MathScore'])
            extra_target_cat = 'MathScoreCat'
                    
        elif target_column == 'WritingScore':
            self.data['WritingScoreCat'] = pd.cut(self.data['WritingScore'], bins=bins, labels=None)
            self.data = self.data.drop(columns=['WritingScore'])
            extra_target_cat = 'WritingScoreCat'
            
        elif target_column == 'ReadingScore':
            self.data['ReadingScoreCat'] = pd.cut(self.data['ReadingScore'], bins=bins, labels=None)
            self.data = self.data.drop(columns=['ReadingScore'])
            extra_target_cat = 'ReadingScoreCat'
            
        self.label_encoders = {}
        categorical_columns = [
            'Gender', 'EthnicGroup', 'ParentEduc', 'LunchType', 'TestPrep',
            'ParentMaritalStatus', 'PracticeSport', 'IsFirstChild', 'TransportMeans',
            'WklyStudyHours', extra_target_cat
        ]
        
        for column in categorical_columns:
            print(f"column: {column}")
            le = LabelEncoder()
            self.data[column] = le.fit_transform(self.data[column])
            self.label_encoders[column] = le

        return self.data, self.label_encoders

class Model:
    """
    Score bins LogisticRegression model
    """
    def __init__(self, data):
        self.data = data
    
    def split_data(self, target_column):
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        return model

class XAI:
    """
    XAI: Explainable AI
    """
    def __init__(self, model, X_train):
        feature_dependence = "independent"
        self.explainer = shap.LinearExplainer(model, X_train, feature_dependence=feature_dependence)
    
    def explain(self, X_test):
        shap_values = self.explainer.shap_values(X_test)
        return shap_values

class DataSecurity:
    def __init__(self, data):
        self.data = data
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
    
    def encrypt_data(self):
        encrypted_data = self.data.applymap(lambda x: self.cipher_suite.encrypt(str(x).encode()).decode())
        return encrypted_data
    
    def decrypt_data(self, encrypted_data):
        decrypted_data = encrypted_data.applymap(lambda x: self.cipher_suite.decrypt(x.encode()).decode())
        return decrypted_data

class ModelEvaluator:
    def __init__(self, model, X_test, y_test, labels):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.labels = labels
    
    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        cr = classification_report(self.y_test, y_pred)

        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(cr)
    
        # Plotting confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.labels, yticklabels=self.labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()


class StudentExamResultFramework():
    def __init__(self, csv_file_path, nb_bins=5):
        self.csv_file_path = csv_file_path
        self.nb_bins = nb_bins
        self.student_data = StudentExamData(self.csv_file_path)
        self.pre_processor = DataPreprocessor(self.student_data.data)

    def run_pipeline(self, target_column):
        """set target column"""
        self.target_column = target_column

        """analyze data"""
        stats = self.analyze_data()
        print("stats: ", stats)

        """clean data"""
        self.cleaned_data, self.label_encoders = self.clean_data()
        target_column = self.target_column + 'Cat'
        self.labels = [str(li) for li in self.label_encoders[target_column].inverse_transform( list(range(self.nb_bins)) )]
        
        """get model, split dataset, train the model"""
        self.model = self.get_model()
        self.X_train, self.X_test, self.y_train, self.y_test = self.model.split_data(target_column)
        self.train_model()

        """evaluate model on test data and plot accuracies"""
        self.evaluator = self.get_evaluator()
        self.eval_model()

        """explain AI"""
        self.x_ai = self.get_xai()
        self.explain_ai()

        """security; encrypt/decrypt data"""
        self.security = self.get_security()
        self.encrypted_data = self.encrypt_data()
        decrypted_data = self.get_decrypted_data(self.encrypted_data)

    def analyze_data(self):
        return self.student_data.analyze_data()

    def clean_data(self):
        cleaned_data, label_encoders = self.pre_processor.clean_data(self.target_column, bins=self.nb_bins)
        return cleaned_data, label_encoders

    def get_model(self):
        model = Model(self.cleaned_data)
        return model

    def train_model(self):
        self.model = self.model.train_model(self.X_train, self.y_train)

    def get_evaluator(self):
        evaluator = ModelEvaluator(self.model, self.X_test, self.y_test, self.labels)
        return evaluator

    def eval_model(self):
        self.evaluator.evaluate_model()

    def get_xai(self):
        x_ai = XAI(self.model, self.X_train)
        return x_ai

    def explain_ai(self):
        shap_values = self.x_ai.explain(self.X_test)
        shap.summary_plot(shap_values, self.X_test, feature_names=self.X_test.columns)

    def get_security(self):
        security = DataSecurity(self.cleaned_data)
        return security
    
    def encrypt_data(self):
        encrypted_data = self.security.encrypt_data()
        return encrypted_data
        print("Data encrypted successfully.")
    
    def get_decrypted_data(self, encrypted_data):
        decrypted_data = self.security.decrypt_data(encrypted_data)
        print("Data decrypted successfully.")
        return decrypted_data



