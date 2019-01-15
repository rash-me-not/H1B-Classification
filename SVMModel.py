import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from statistics import mode
from sklearn import preprocessing, svm
import matplotlib.pyplot as plt
from util import func_confusion_matrix

def SVMModel(train_x,train_y,val_x,val_y,testX,testY):
    #Building the SVM model, chosen c = 4 as best value for the model.
    score=0
    best_kernel = 'linear'
    kernel_types = ['linear', 'poly', 'rbf']
    svm_kernel_error = []
    for kernel_value in kernel_types:
        model = svm.SVC(kernel=kernel_value, C=5)
        model.fit(X=train_x, y=train_y)
        score = model.score(val_x, val_y)
        svm_kernel_error.append(1-(score))
		
    print("Predict the score for the training data set")
    model = svm.SVC(kernel=best_kernel, C=5)
    model.fit(X=train_x, y=train_y)
    score = model.score(val_x, val_y)
    print("ScorePrinted:",score)
    print("Presenting results for test data set")
    y_pred = model.predict(testX)
    #testY = Y_test.values
    #print("Accuracy:",metrics.accuracy_score(testY, y_pred))
    conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(testY, y_pred)

    print("Confusion Matrix: ")
    print(conf_matrix)
    print("Average Accuracy: {}\n".format(accuracy))
    print("Per-Class Precision: {}]\n".format(precision_array))
    print("Per-Class Recall: {}".format(recall_array))
    return svm_kernel_error,(accuracy*100),(max(recall_array)*100),(max(precision_array)*100)