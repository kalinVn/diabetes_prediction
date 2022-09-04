from diabetes_prediction_custom.App import App as AppCustom
from diabetes_prediction_keras.App import App as AppKeras
from diabetes_prediction_keras.Visualizator import Visualizator
import pandas as pd

CSV_FILE_PATH = 'dataset/diabetes.csv'

input_data = [9, 102, 76, 37, 0, 32.9, 0.665, 46]


def diabetes_prediction_custom():
    app = AppCustom(CSV_FILE_PATH)
    app.standardize_data()
    app.build()
    app.test_accuracy_score()

    app.predict(input_data)


def diabetes_prediction_keras():
    df = pd.read_csv('dataset/diabetes.csv')
    visualizator = Visualizator(df)
    app = AppKeras(df)
    app.replace_zero_column()
    app.fill_null_columns()
    app.standard_data()
    app.separate_data()
    app.build()
    app.test_accuracy()

    patient_results = [9, 102, 76, 37, 0, 32.9, 0.665, 46]
    patient_data = app.get_patient_data(patient_results)
    app.predict(patient_data)

    c_matrix = app.get_confusion_matrix(patient_data)
    # visualizator.show_confusion_matrix(c_matrix)
    visualizator.show_hist_plot_dataset(df, x='Glucose')

# diabetes_prediction_custom()


diabetes_prediction_keras()


