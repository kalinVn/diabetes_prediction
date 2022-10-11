import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt


class Visualizator:

    def __init__(self, df):
        self.df = df
        self.sns = sns

    def show_confusion_matrix(self, c_matrix):
        ax = self.sns.heatmap(c_matrix, annot=True,
                              xticklabels=['No Diabetes', 'Diabetes'],
                              yticklabels=['No Diabetes', 'Diabetes'],
                              cbar=False, cmap='Blues')
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Actual")
        plt.show()

    def show_hist_plot_dataset(self, diabetes_dataset, x):
        self.sns.histplot(x=x, data=diabetes_dataset, color='red')
        plt.show()

    def roc_curve(self, frp, tpr):
        plt.plot(frp, tpr)

        # diagonal line
        plt.plot([0, 1], [0, 1], '--', color='black')

        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()


