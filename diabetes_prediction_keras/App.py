import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix


class App:

    def __init__(self, df):
        self.model = None
        self.df = df
        self.df_scaled = None
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_val = None
        self.y_val = None

    def replace_zero_column(self):
        self.df['Glucose'] = self.df['Glucose'].replace(0, np.nan)
        self.df['BloodPressure'] = self.df['BloodPressure'].replace(0, np.nan)
        self.df['Pregnancies'] = self.df['Pregnancies'].replace(0, np.nan)
        self.df['SkinThickness'] = self.df['SkinThickness'].replace(0, np.nan)
        self.df['Insulin'] = self.df['Insulin'].replace(0, np.nan)
        self.df['BMI'] = self.df['BMI'].replace(0, np.nan)
        self.df['DiabetesPedigreeFunction'] = self.df['DiabetesPedigreeFunction'].replace(0, np.nan)
        self.df['Age'] = self.df['Age'].replace(0, np.nan)

    def fill_null_columns(self):
        # print(self.df.colums)
        self.df['Glucose'] = self.df['Glucose'].fillna(self.df['Glucose'].mean())
        self.df['BloodPressure'] = self.df['BloodPressure'].fillna(self.df['BloodPressure'].mean())
        self.df['Pregnancies'] = self.df['Pregnancies'].fillna(self.df['Pregnancies'].mean())
        self.df['SkinThickness'] = self.df['SkinThickness'].fillna(self.df['SkinThickness'].mean())
        self.df['Insulin'] = self.df['Insulin'].fillna(self.df['Insulin'].mean())
        self.df['BMI'] = self.df['BMI'].fillna(self.df['BMI'].mean())
        self.df['DiabetesPedigreeFunction'] = self.df['DiabetesPedigreeFunction'].fillna(
            self.df['DiabetesPedigreeFunction'].mean())
        self.df['Age'] = self.df['Age'].fillna(self.df['Age'].mean())

    def standard_data(self):
        df_scaled = preprocessing.scale(self.df)
        self.df_scaled = pd.DataFrame(df_scaled, columns=self.df.columns)
        self.df_scaled['Outcome'] = self.df['Outcome']

        self.x = self.df.loc[:, self.df.columns != 'Outcome']
        self.y = self.df.loc[:, 'Outcome']

    def separate_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.2)

    def build(self):
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_dim=8))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train, epochs=200)

    def test_accuracy(self):
        scores = self.model.evaluate(self.x_train, self.y_train)
        print("Training accuracy: %.2f%%\n" % scores[1] * 100)

        scores = self.model.evaluate(self.x_test, self.y_test)
        print("Testing accuracy: %.2f%%\n" % scores[1] * 100)

    def predict(self, patient_data):
        y_test_predict = self.model.predict(patient_data)
        # classes_x = (self.model.predict(self.x_test) > 0.5).astype("int32")
        classes_x = (self.model.predict(patient_data) > 0.5).astype("int32")
        # print(classes_x)
        if classes_x[0][0] == 1:
            print('Patient is not diabetic')
        else:
            print('Patient is diabetic')

    def get_prediction_probs(self):

        return self.model.predict(self.x_test)

    def get_patient_data(self, patient_results):
        self.df = pd.DataFrame(
            [patient_results],
            columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                     'DiabetesPedigreeFunction', 'Age'])

        return self.df

    def get_confusion_matrix(self, y_prediction):
        # y_test_predict = self.get_prediction(y_prediction)
        # y_test_predict = self.model.predict(self.x_test)
        # classes_x = np.argmax(y_test_predict, axis=1)
        classes_x = (self.model.predict(self.x_test) > 0.5).astype("int32")
        c_matrix = confusion_matrix(self.y_test, classes_x)

        return c_matrix

    def get_x_test(self):
        return self.x_test

    def get_y_test(self):
        return self.y_test

