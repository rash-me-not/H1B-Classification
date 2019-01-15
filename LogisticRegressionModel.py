import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,roc_curve,roc_auc_score
from statistics import mode
#from sklearn import preprocessing, svm
import matplotlib.pyplot as plt
from util import func_confusion_matrix


def LogisticRegressionModel(train_x,train_y,val_x,val_y,testX,testY):
    C_param_range = [0.001, 0.01, 0.1, 1, 10, 100]

    score = []
    accuracy = []
    y_pred_all =[]
    for i in C_param_range:

        clf = LogisticRegression()
        # call the function fit() to train the class instance
        clf.fit(train_x, train_y)
        # scores over testing samples
        score.append(clf.score(val_x, val_y))
        y_pred = clf.predict(testX)
        y_pred_all.append(y_pred)
        accuracy.append(accuracy_score(testY,y_pred))
    #accuracy,precision,recall,f1_score=func_calConfusionMatrix(y_pred,testY)

    best_c = np.argmax(accuracy)
    y_pred_best = y_pred_all[best_c]
    conf_matrix,accuracy,recall_array,precision_array=func_confusion_matrix(testY, y_pred_best)
	
    print("Confusion Matrix: {} \n".format(conf_matrix))
    print("Accuracy with the test data: {} \n".format(accuracy))
    print("Per-Class Precision is: {} \n".format(precision_array))
    print("Per-Class Recall rate: {} \n".format(recall_array))

    logit_roc_auc = roc_auc_score(testY, y_pred_best)
    fpr, tpr, thresholds = roc_curve(testY, clf.predict_proba(testX)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    return (accuracy*100),(max(recall_array)*100),(max(precision_array)*100)

