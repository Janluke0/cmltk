#!/usr/bin/env python3
from logistic_regression import LogisticRegression
import matrix  as mat 
import numpy as np

if __name__ == '__main__':
    import sklearn.datasets as ds 
    X, y = ds.load_breast_cancer(return_X_y = True)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
    model = LogisticRegression()
    X_train, y_train = mat.Matrix(X_train),  mat.Matrix(y_train)
    it = model.fit(X_train, y_train, max_it=10000, tol=10e-8)
    print("#it",it )
    X_test, y_test = mat.Matrix(X_test),  mat.Matrix(y_test)
    pred = model.predict(X_train)
    print("train accurancy", (np.asarray(pred)==np.asarray(y_train)).mean())
    pred = model.predict(X_test)
    print("test accurancy", (np.asarray(pred)==np.asarray(y_test)).mean())