from sklearn.metrics import roc_curve
from visualization import hist, show_roc_curve, confusion_matrix, dense_plot, hist_plot, scatter_plot, line_plot
from diabetes_prediction_custom.App import App as AppCustom
from diabetes_prediction_keras.App import App as AppKeras
import pandas as pd

CSV_FILE_PATH = 'dataset/diabetes.csv'

input_data = [9, 102, 76, 37, 0, 32.9, 0.665, 46]


def diabetes_prediction_custom():
    df = pd.read_csv('dataset/diabetes.csv')
    app = AppCustom(CSV_FILE_PATH)
    app.standardize_data()
    app.build()
    app.test_accuracy_score()
    app.predict(input_data)

    # params = dict(x='Pregnancies', data=df, binwidth=1)
    # hist_plot(df)
    # scatter_plot(df)
    # line_plot(df)


def diabetes_prediction_keras(visulize=False):
    df = pd.read_csv('dataset/diabetes.csv')
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
    # confusion_matrix(c_matrix)
    # hist(df)
    # dense_plot(df)
    y_test_prediction_probs = app.get_prediction_probs()
    y_test = app.get_y_test()
    frp, trp, _ = roc_curve(y_test, y_test_prediction_probs)

    if visulize:
        confusion_matrix(c_matrix)
        hist(df)
        dense_plot(df)
        show_roc_curve(frp, trp)
def visualize():
    df = pd.read_csv('dataset/diabetes.csv')


# diabetes_prediction_custom()


diabetes_prediction_keras(True)


