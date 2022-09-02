from diabetes_prediction_custom.App import App as AppCustom
from diabetes_prediction_keras.App import App as AppKeras

CSV_FILE_PATH = 'dataset/diabetes.csv'

input_data = [9, 102, 76, 37, 0, 32.9, 0.665, 46]


def diabetes_prediction_custom():
    app = AppCustom(CSV_FILE_PATH)
    app.standardize_data()
    app.build()
    app.test_accuracy_score()

    app.predict(input_data)


def diabetes_prediction_keras():

    app = AppKeras(CSV_FILE_PATH)
    app.replace_zero_column()
    app.fill_null_columns()
    app.standard_data()
    app.separate_data()
    app.build()

    patient_data = app.get_patient_data(input_data)
    app.predict(patient_data)


# diabetes_prediction_custom()
diabetes_prediction_keras()


