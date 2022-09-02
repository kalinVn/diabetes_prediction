import numpy as np
import pandas as pd
from diabetes_prediction_custom.ClassifierFactory import ClassifierFactory
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class App:

    def __init__(self, csv_path):
        self.standard_data = None
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.csv_path = csv_path
        self.diabetes_dataset = pd.read_csv(csv_path)
        self.scaler = StandardScaler()
        self.classifierFactory = ClassifierFactory()
        self.classifier = self.classifierFactory.get_classifier_by_name(name="svm")

    def standardize_data(self):
        self.x = self.diabetes_dataset.drop(columns='Outcome', axis=1)
        self.y = self.diabetes_dataset['Outcome']

        self.scaler.fit(self.x)
        self.standard_data = self.scaler.transform(self.x)

    def build(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                                                stratify=self.y, random_state=2)
        self.classifier.fit(self.x_train, self.y_train)

    def test_accuracy_score(self):
        x_train_prediction = self.classifier.predict(self.x_train)
        training_data_accuracy = accuracy_score(x_train_prediction, self.y_train)
        print('Accuracy score of the training data: ', training_data_accuracy)

        x_test_prediction = self.classifier.predict(self.x_test)
        test_data_accuracy = accuracy_score(x_test_prediction, self.y_test)
        print('Accuracy score of the test data: ', test_data_accuracy)

    def predict(self, data):
        input_data_as_numpy_array = np.asarray(data)
        print(data)
        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # standardize the input data
        std_data = self.scaler.transform(input_data_reshaped)

        prediction = self.classifier.predict(std_data)
        # print(prediction)

        if prediction[0] == 0:
            print('Print the person is not diabetic')
        else:
            print('Print the person is diabetic')