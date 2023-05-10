import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree,export_graphviz
)
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import pydotplus


def calculate_many_metrics(y_test, y_pred, y_proba=None, print_to_console=False):
    accuracy = accuracy_score(y_test, y_pred).round(3)
    precision = precision_score(y_test, y_pred, average='macro').round(3)
    recall = recall_score(y_test, y_pred, average='macro').round(3)
    f_scores = f1_score(y_test, y_pred, average='macro').round(3)
    mcc = matthews_corrcoef(y_test, y_pred).round(3)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred).round(3)
    if print_to_console:
        print("accuracy = ", accuracy)
        print("precision =", precision)
        print("recall = ", recall_score(y_test, y_pred, average='macro').round(3))
        print("f_scores = ", f1_score(y_test, y_pred, average='macro').round(3))
        print("mcc = ", matthews_corrcoef(y_test, y_pred).round(3))
        print("balanced_accuracy = ", balanced_accuracy_score(y_test, y_pred).round(3))

    # Help here
    roc_auc_stat = None
    if y_proba is not None:
        roc_auc_stat = roc_auc_score(y_test, y_proba, multi_class='ovr')
        if print_to_console:
            print('ROC-AUC Stat:', roc_auc_stat.round(3))

    if print_to_console:
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred).round(3))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred).round(3))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f_scores": f_scores,
        "mcc": mcc,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc_stat": roc_auc_stat
    }


def vizualizator(y_vector, text, metrica, train_metrica=None):
    plt.plot(y_vector, metrica, label ='Test')
    if train_metrica is not None:
        plt.plot(y_vector, train_metrica, label ='Train')
    plt.title(f"{text} при різній кількості сусідів")
    plt.xlabel("n_neighbors")
    plt.ylabel("Score")
    plt.legend()
    plt.grid()
    plt.show()


