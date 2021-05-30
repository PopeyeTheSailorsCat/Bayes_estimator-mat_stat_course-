from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from data_collector import get_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_class(dataset):
    dataset.isnull().sum()

    bins = (2, 6.5, 8)
    labels = ['bad', 'good']
    dataset['quality'] = pd.cut(x=dataset['quality'], bins=bins, labels=labels)

    print(dataset['quality'].value_counts())
    dataset['quality'].value_counts()

    from sklearn.preprocessing import LabelEncoder

    labelencoder_y = LabelEncoder()
    dataset['quality'] = labelencoder_y.fit_transform(dataset['quality'])
    print(dataset.head())

    corr = dataset.corr()
    # Plot figsize
    fig, ax = plt.subplots(figsize=(10, 8))
    # Generate Heat Map, allow annotations and place floats in map
    sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f")
    # Apply xticks
    plt.xticks(range(len(corr.columns)), corr.columns)
    # Apply yticks
    plt.yticks(range(len(corr.columns)), corr.columns)
    # show plot
    plt.show()

    dataset['quality'].value_counts()

    X = dataset.drop('quality', axis=1).values
    y = dataset['quality'].values.reshape(-1, 1)

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Shape of X_train: ", X_train.shape)
    print("Shape of X_test: ", X_test.shape)
    print("Shape of y_train: ", y_train.shape)
    print("Shape of y_test", y_test.shape)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    # Fitting classifier to the Training set
    from sklearn.naive_bayes import GaussianNB

    classifier_nb = GaussianNB()
    classifier_nb.fit(X_train_scaled, y_train.ravel())

    # Predicting Cross Validation Score
    cv_nb = cross_val_score(estimator=classifier_nb, X=X_train_scaled, y=y_train.ravel(), cv=10)
    print("CV: ", cv_nb.mean())

    y_pred_nb_train = classifier_nb.predict(X_train_scaled)
    accuracy_nb_train = accuracy_score(y_train, y_pred_nb_train)
    print("Training set: ", accuracy_nb_train)
    print(y_pred_nb_train)
    y_pred_nb_test = classifier_nb.predict(X_test_scaled)
    accuracy_nb_test = accuracy_score(y_test, y_pred_nb_test)
    print("Test set: ", accuracy_nb_test)

    confusion_matrix(y_test, y_pred_nb_test)

    tp_nb = confusion_matrix(y_test, y_pred_nb_test)[0, 0]
    fp_nb = confusion_matrix(y_test, y_pred_nb_test)[0, 1]
    tn_nb = confusion_matrix(y_test, y_pred_nb_test)[1, 1]
    fn_nb = confusion_matrix(y_test, y_pred_nb_test)[1, 0]
    return tp_nb, fp_nb, tn_nb, fn_nb, accuracy_nb_train, accuracy_nb_test, cv_nb
