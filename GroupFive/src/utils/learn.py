import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    good = np.sum(y_true == y_pred)
    return good / y_true.shape[0]


def evaluate_model(model_creator, X, y, n_sim=100):
    accuracy = np.zeros(n_sim)
    for i in range(n_sim):
        model = model_creator()
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # print(y_test, y_pred)
        accuracy[i] = accuracy_score(np.array(y_test), np.array(y_pred))
    return accuracy
