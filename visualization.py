import matplotlib
import seaborn as sns
import warnings
from matplotlib import pyplot as plt
matplotlib.use("TkAgg")
warnings.filterwarnings("ignore")


def hist(df):
    df.hist()
    plt.tight_layout()
    plt.show()


def show_roc_curve(frp, tpr):
    plt.plot(frp, tpr)

    # diagonal line
    plt.plot([0, 1], [0, 1], '--', color='black')

    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


def confusion_matrix(c_matrix):
    ax = sns.heatmap(c_matrix, annot=True,
                          xticklabels=['No Diabetes', 'Diabetes'],
                          yticklabels=['No Diabetes', 'Diabetes'],
                          cbar=False, cmap='Blues')
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    plt.show()


def dense_plot(df):
    plt.subplots(3, 3, figsize=(20, 20))

    for idx, col in enumerate(df.columns):
        ax = plt.subplot(3, 3, idx + 1)
        ax.yaxis.set_ticklabels([])
        sns.distplot(df.loc[df.Outcome == 0][col], hist=False, axlabel=False,
                     kde_kws={'linestyle': '-', 'color': 'black', 'label': "No Diabetes"})
        sns.distplot(df.loc[df.Outcome == 1][col], hist=False, axlabel=False,
                     kde_kws={'linestyle': '--', 'color': 'black', 'label': "Diabetes"})
        ax.set_title(col)

    plt.subplot(3, 3, 9).set_visible(False)
    plt.tight_layout()
    plt.show()

