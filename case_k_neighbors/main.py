# 0. Import all libs
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import pydotplus
from utils import calculate_many_metrics, vizualizator
import warnings
warnings.filterwarnings("ignore")

# 1. Відкрити та зчитати наданий файл з даними.
my_variant = 8
datafile_path = "data/WQ-R.csv"
df = pd.read_csv(datafile_path, sep=";")

# 2. Визначити та вивести кількість записів.
print(f'df shape {df.shape}')

# 3. Вивести атрибути набору даних.
print('List of available attributes:\n', list(df.columns))
# make y, X
y = df["quality"]
X = df.drop(["quality"], axis=1)
print('Distribution of quality:\n', y.value_counts(dropna=False).sort_index()/len(df))

# 4 Отримати десять варіантів перемішування набору даних та розділення його на навчальну (тренувальну)
# та тестову вибірки, використовуючи функцію ShuffleSplit.
shuffle_split = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

# Сформувати начальну та тестові вибірки наоснові восьмого варіанту. З’ясувати збалансованість набору даних.
for i, (train_index, test_index) in enumerate(shuffle_split.split(X)):
    if i == my_variant - 1:
        print(f"Fold {i}:")
        print(f"  Train: index={train_index[0]}:{train_index[-1]}")
        print(f"  Test:  index={test_index[0]}:{test_index[-1]}")

        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        print('Distribution of quality in train:\n', y_train.value_counts(dropna=False).sort_index() / len(y_train))
        print('Distribution of quality in test:\n', y_test.value_counts(dropna=False).sort_index() / len(y_test))

# 5. Використовуючи функцію KNeighborsClassifier бібліотеки scikit-learn,
# збудувати класифікаційну модель на основі методу k найближчих сусідів
# (значення всіх параметрів залишити за замовчуванням) та навчити її на тренувальній вибірці,
# вважаючи, що цільова характеристика визначається стовпчиком quality,
# а всі інші виступають в ролі вихідних аргументів.
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
print(f"Train accuracy, calculated by knn.score method: {round(accuracy, 4)}")

# 6. Обчислити класифікаційні метрики збудованої моделі для тренувальної та тестової вибірки.
# Представити результати роботи моделі на тестовій вибірці графічно.
n_neighbors_train_stats = {}
n_neighbors_test_stats = {}
n_neighbors_iterations = 20

for i in range(n_neighbors_iterations):
    print(f"\n Number of neighbors = {i + 1}")
    print('Model training started ...')
    knn = KNeighborsClassifier(n_neighbors=i + 1)
    knn.fit(X_train, y_train)

    print(f"\n Train")
    y_proba_train = knn.predict_proba(X_train)
    y_pred_train = knn.predict(X_train)
    train_metrics = calculate_many_metrics(y_train, y_pred_train, y_proba_train, print_to_console=True)
    n_neighbors_train_stats[i] = train_metrics

    print(f"\n Test")
    y_proba = knn.predict_proba(X_test)
    y_pred = knn.predict(X_test)
    test_metrics = calculate_many_metrics(y_test, y_pred, y_proba, print_to_console=True)
    n_neighbors_test_stats[i] = test_metrics

    print("\n ---------------------------------")

n_neighbors_train_stats = pd.DataFrame(n_neighbors_train_stats).T
n_neighbors_test_stats = pd.DataFrame(n_neighbors_test_stats).T

# 7. З’ясувати вплив кількості сусідів (від 1 до 20) на результати класифікації. Результати представити графічно.
y_vector = list(range(1, n_neighbors_iterations+1))
