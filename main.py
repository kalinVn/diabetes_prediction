import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

CSV_FILE_PATH = 'dataset/diabetes.csv'

scaler = StandardScaler()
classifier = svm.SVC(kernel='linear')


def init():
    diabetes_dataset = pd.read_csv(CSV_FILE_PATH)

    # Separating the data and labels
    x = diabetes_dataset.drop(columns='Outcome', axis=1)
    y = diabetes_dataset['Outcome']

    standardize_data(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
    # training the support vector machine classifier
    classifier.fit(x_train, y_train)

    test_accuracy_score(classifier, x_train, y_train, x_test, y_test)


def standardize_data(x):
    scaler.fit(x)
    stand_data = scaler.transform(x)


def get_classifier():
    return svm.SVC(kernel='linear')



def test_accuracy_score(classifier ,x_train, y_train, x_test, y_test):
    # Accuracy Score on the training data
    x_train_prediction = classifier.predict(x_train)
    training_data_accuracy = accuracy_score(x_train_prediction, y_train)
    print('Accuracy score of the training data: ', training_data_accuracy)

    # Accuracy Score on the test data
    x_test_prediction = classifier.predict(x_test)
    test_data_accuracy = accuracy_score(x_test_prediction, y_test)
    print('Accuracy score of the test data: ', test_data_accuracy)


def predict(data):
    input_data_as_numpy_array = np.asarray(data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # standardize the input data
    std_data = scaler.transform(input_data_reshaped)

    prediction = classifier.predict(std_data)
    # print(prediction)

    if prediction[0] == 0:
        print('Print the person is not diabetic')
    else:
        print('Print the person is diabetic')


init()

input_data = (1, 85, 66, 29, 0, 26.6, 0.351, 31)
predict(input_data)

