# -*- coding: utf-8 -*-
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
from DataPreProcessing import merge_labels,fill_nan_values,prevailing_wage,case_submission_year_range,classify_employer,preprocessingTrainingdata,preprocessingTestingdata
from LogisticRegressionModel import LogisticRegressionModel
from SVMModel import SVMModel
from NeuralNetworksModel import feed_forward_model
import time






def drawResults():
    fig,ax=fig, ax = plt.subplots()
    index=np.arange(len(accuracyList))
    print(index)
    width = 0.35  # the width of the bars
    rects1 = ax.bar(index - width/2, accuracyList, width,
                color='SkyBlue', label='Accuracy')
    rects2 = ax.bar(index + width/2, precisionList, width,
                color='Red', label='Precision')
    ax.set_ylabel('Scores')
    ax.set_title('Performance from different models')
    ax.set_xticks(index)
    ax.set_xticklabels(('Logistic Regression', 'SVM','Neural NetWorks'))
    ax.legend()
    plt.show()

def	drawkernelError(kernel_error_rate):
    kernel_Types=['linear','poly','rbf']
    plt.plot(kernel_Types, kernel_error_rate,marker='o')
    plt.title('SVM by Kernels')
    plt.xlabel('Kernel')
    plt.ylabel('error')
    plt.xticks(kernel_Types)
    plt.show()

def timeGraph(runtimes):
    models=['LogisticRegression','SVM','Neural Networks']
    plt.plot(models,runtimes,marker='o',color='red')
    plt.title('Time graph for different models')
    plt.ylabel('Time in seconds')
    plt.xlabel('Models')
    plt.xticks(models)
    plt.show()
 
def certifiedandDeniedGraphTrainingData():
    print('Plotting Case Status graph for Training Data')
    df_train = pd.read_csv('File 1 - H1B Dataset.csv',encoding="ISO-8859-1")
    merge_labels(df_train)
    fill_nan_values(df_train)
    prevailing_wage(df_train)
    case_submission_year_range(df_train)
    classify_employer(df_train)
    certified = df_train[df_train['CASE_STATUS']=='CERTIFIED']
    certified = certified['CASE_STATUS'].count()
    denied=df_train[df_train['CASE_STATUS']=='DENIED']
    denied=denied['CASE_STATUS'].count()
    
    df=pd.DataFrame({'Training Data':['Certified','Denied'],'Number of Petitions':[certified,denied]})
    fig,ax = plt.subplots()
    ax=df.plot.bar(x='Training Data',y='Number of Petitions',rot=0,legend=True)
    plt.title('Case Status vs Number of Petitions')
    plt.show()

def certifiedandDeniedGraphTestData():
    print('Plotting Case Status graph for Test Data')
    df_train = pd.read_csv('File 2 - H1B Dataset.csv',encoding="ISO-8859-1")
    merge_labels(df_train)
    fill_nan_values(df_train)
    prevailing_wage(df_train)
    case_submission_year_range(df_train)
    classify_employer(df_train)
    certified = df_train[df_train['CASE_STATUS']=='CERTIFIED']
    certified = certified['CASE_STATUS'].count()
    denied=df_train[df_train['CASE_STATUS']=='DENIED']
    denied=denied['CASE_STATUS'].count()
    
    df=pd.DataFrame({'Training Data':['Certified','Denied'],'Number of Petitions':[certified,denied]})
    fig,ax = plt.subplots()
    ax=df.plot.bar(x='Training Data',y='Number of Petitions',rot=0,legend=True)
    plt.title('Case Status vs Number of Petitions')
    plt.show()

if __name__ == "__main__":
    accuracyList=[]
    precisionList=[]
    recallList=[]
    isLogistic=False
    isSVM=False
    isNeural=False
    runtimes=[]
    
    while True:
        print("Select the option below:")
        print('''(1) Logistic Regression\n (2) Support Vector Machine\n (3) Feed Neural Networks\n (4) To generate graphs\n (5) To exit''')
        #myvar = easygui.enterbox("What, is your favorite color?")
        user_input=input('Enter your choice:')
        
        if(user_input=='1'):
            isLogistic=True
            
            print("Logistic Regression Model selected\n")
            train_x,train_y,val_x,val_y=preprocessingTrainingdata(user_input)
            testX,testY=preprocessingTestingdata(user_input)
            start_time=time.time()
            accuracy,recall,precision=LogisticRegressionModel(train_x,train_y,val_x,val_y,testX,testY)
            runtimes.append(time.time()-start_time)
            print(runtimes)
            accuracyList.append(accuracy)
            precisionList.append(precision)
            recallList.append(recall)
        elif(user_input=='2'):
            isSVM=True
            print("SVM Model selected")
            train_x,train_y,val_x,val_y=preprocessingTrainingdata(user_input)
            testX,testY=preprocessingTestingdata(user_input)
            start_time=time.time()
            error_rate_kernel,accuracy,recall,precision=SVMModel(train_x,train_y,val_x,val_y,testX,testY)
            runtimes.append(time.time()-start_time)
            print(runtimes)
            accuracyList.append(accuracy)
            precisionList.append(precision)
            recallList.append(recall)
            print(error_rate_kernel)
            drawkernelError(error_rate_kernel)
        elif(user_input=='3'):
            isNeural=True
            print('Neural Networks selected')
            train_x,train_y,val_x,val_y=preprocessingTrainingdata(user_input)
            testX,testY=preprocessingTestingdata(user_input)
            start_time=time.time()
            accuracy,recall,precision=feed_forward_model(train_x,train_y,val_x,val_y,testX,testY)
            print(runtimes)
            runtimes.append(time.time()-start_time)
            accuracyList.append(accuracy)
            precisionList.append(precision)
        elif(user_input=='4'):
            if(isLogistic&isSVM&isNeural):
                drawResults()
                timeGraph(runtimes)
                certifiedandDeniedGraphTrainingData()
                certifiedandDeniedGraphTestData()
            else:
                print("Please run all the models before generating the graphs \n \n")
                
        else:
            if(user_input=='5'):
                print("Thanks for using the program good byee!!")
            else:    
                print("Not a valid input, program is exiting bye!!")
            break
    
    



        