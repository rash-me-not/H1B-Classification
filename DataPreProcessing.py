import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from statistics import mode


def case_submission_year_range(df):
    df['CASE_SUBMITTED_YEAR_RANGE'] = np.nan
    for i in range(len(df['CASE_SUBMITTED_YEAR'])):
        if int(df['CASE_SUBMITTED_YEAR'][i]) <= 2012:
            df.loc[i, 'CASE_SUBMITTED_YEAR_RANGE'] = 'BEFORE 2012'
        if int(df['CASE_SUBMITTED_YEAR'][i]) > 2012:
            df.loc[i, 'CASE_SUBMITTED_YEAR_RANGE']  = 'AFTER 2012'
			
def prevailing_wage(df):
    df['PREVAILING_WAGE_RANGE'] = np.nan
    for i in range(len(df['PREVAILING_WAGE'])):
        if int(df['PREVAILING_WAGE'][i]) <= 20000:
            df.loc[i, 'PREVAILING_WAGE_RANGE'] = '0 - 20000'
        if int(df['PREVAILING_WAGE'][i]) > 20000 and int(df['PREVAILING_WAGE'][i]) <= 50000:
            df.loc[i, 'PREVAILING_WAGE_RANGE']  = '20000 - 50000'
        if int(df['PREVAILING_WAGE'][i]) > 50000 and int(df['PREVAILING_WAGE'][i]) <= 120000:
            df.loc[i, 'PREVAILING_WAGE_RANGE'] = '50000 - 120000'
        if int(df['PREVAILING_WAGE'][i]) > 120000 and int(df['PREVAILING_WAGE'][i]) <= 250000:
            df.loc[i, 'PREVAILING_WAGE_RANGE'] = '120000 - 250000'
        if int(df['PREVAILING_WAGE'][i]) > 250000:
            df.loc[i, 'PREVAILING_WAGE_RANGE'] ='>250000'

def merge_labels(df):
    '''Merges labels like CERTIFIEDWITHDRAWN in the dataset into CERTIFIED class, and REJECTED, INVALIDATED with DENIED class
    Here we are transforming multiclass dataset into binary classification data with labels CERTIFIED and DENIED'''
    df['CASE_STATUS'] = df['CASE_STATUS'].replace(['CERTIFIEDWITHDRAWN'], ['CERTIFIED'])
    df['CASE_STATUS'] = df['CASE_STATUS'].replace(['REJECTED'], ['DENIED'])
    df['CASE_STATUS'] = df['CASE_STATUS'].replace(['INVALIDATED'], ['DENIED'])
    labels = df['CASE_STATUS'].unique()
    #print(labels)


def fill_nan_values(df):
    '''The dataset consists of many nan values. These are replaced by the mode for various columns like EMPLOYER_NAME,
    EMPLOYER_STATE, FULL_TIME_POSITION ,PW_UNIT_OF_PAY ,PW_SOURCE, PW_SOURCE_YEAR, H-1B_DEPENDENT, WILLFUL_VIOLATOR. For the column
    PREVAILING_WAGE we replace the nan columns with the mean value of the wage data. Also, if the SOC_NAME () is not available,
     we replace it with hardcoded value Others'''

    df['EMPLOYER_NAME'] = df['EMPLOYER_NAME'].fillna(df['EMPLOYER_NAME'].mode()[0])
    df['EMPLOYER_STATE'] = df['EMPLOYER_STATE'].fillna(df['EMPLOYER_STATE'].mode()[0])
    df['FULL_TIME_POSITION'] = df['FULL_TIME_POSITION'].fillna(df['FULL_TIME_POSITION'].mode()[0])
    df['PW_UNIT_OF_PAY'] = df['PW_UNIT_OF_PAY'].fillna(df['PW_UNIT_OF_PAY'].mode()[0])
    df['PW_SOURCE'] = df['PW_SOURCE'].fillna(df['PW_SOURCE'].mode()[0])
    df['PW_SOURCE_YEAR'] = df['PW_SOURCE_YEAR'].fillna(df['PW_SOURCE_YEAR'].mode()[0])
    df['H-1B_DEPENDENT'] = df['H-1B_DEPENDENT'].fillna(df['H-1B_DEPENDENT'].mode()[0])
    df['WILLFUL_VIOLATOR'] = df['WILLFUL_VIOLATOR'].fillna(df['WILLFUL_VIOLATOR'].mode()[0])


    df['SOC_NAME'] = df.SOC_NAME.replace(np.nan, 'Others', regex=True)

    df.PREVAILING_WAGE.fillna(df.PREVAILING_WAGE.mean(), inplace=True)


def classify_employer(df):
    # Check if the employer name is a 'university'. Since employers with university in their name have more chances of visa approval
    df['UNIV_EMPLOYER'] = np.nan
    df['EMPLOYER_NAME'] = df['EMPLOYER_NAME'].str.lower()
    df.loc[df['EMPLOYER_NAME'].str.contains('university'), 'UNIV_EMPLOYER'] = 'university'
    df['UNIV_EMPLOYER'] = df.UNIV_EMPLOYER.replace(np.nan, 'non university', regex=True)

    # Broadly classifying the occupations for people filing the visa petition
    df['OCCUPATION'] = np.nan
    df['SOC_NAME'] = df['SOC_NAME'].str.lower()

    df.loc[df['SOC_NAME'].str.contains(
        'computer|graphic|web|developer|programmer|software|it|database|analyst'), 'OCCUPATION'] = 'IT Industry'
    df.loc[df['SOC_NAME'].str.contains(
        'business|managers|planners|management|public relation|executives|supervisor|curator|human resources'), 'OCCUPATION'] = 'Management'
    df.loc[df['SOC_NAME'].str.contains('math|statistic|stats'), 'OCCUPATION'] = 'Maths Department'
    df.loc[df['SOC_NAME'].str.contains('promotion|market|advertis'), 'OCCUPATION'] = 'Marketing Department'
    df.loc[df['SOC_NAME'].str.contains('accountant|finance|acc'), 'OCCUPATION'] = 'Finance Department'
    df.loc[df['SOC_NAME'].str.contains(
        'education|prof|teacher|linguist|teach|counsel|coach'), 'OCCUPATION'] = 'Education Department'
    df.loc[df['SOC_NAME'].str.contains(
        'scientist|science|psychia|doctor|surgeon|biolog|clinical reasearch|physician|dentist|health'), 'OCCUPATION'] = 'Advance Sciences'
    df.loc[
        df['SOC_NAME'].str.contains(
            'engineer|technician|surveyor|architec'), 'OCCUPATION'] = 'Engineering and Architecture'
    df['OCCUPATION'] = df.OCCUPATION.replace(np.nan, 'Others', regex=True)

def preprocessingTrainingdata(user_input):
    print("Pre Processing data for training set")
    if(user_input=='1' or user_input=='3' or user_input=='2'):
        df_train = pd.read_csv('File 1 - H1B Dataset.csv',encoding="ISO-8859-1")

    merge_labels(df_train)

    #clean data by filling the NAN data with appropriate values
    fill_nan_values(df_train)
    prevailing_wage(df_train)
    case_submission_year_range(df_train)
    
    #Create new column OCCUPATION to broadly classify the occupations for H1B petition filers.
    #Also creating column UNIV_EMPLOYER for checking if the emplyer name is a university
    classify_employer(df_train)

    class_mapping = {'CERTIFIED': 0, 'DENIED': 1}
    df_train["CASE_STATUS"] = df_train["CASE_STATUS"].map(class_mapping)

#Creating a copy of dataset for prediction
    df1_train_set = df_train[
    ['FULL_TIME_POSITION', 'PREVAILING_WAGE_RANGE', 'CASE_SUBMITTED_YEAR_RANGE', 'UNIV_EMPLOYER', 'OCCUPATION', 'WORKSITE_STATE',
     'CASE_STATUS']].copy()

    df1_train_set[['FULL_TIME_POSITION', 'PREVAILING_WAGE_RANGE', 'CASE_SUBMITTED_YEAR_RANGE', 'UNIV_EMPLOYER', 'OCCUPATION', 'WORKSITE_STATE',
     'CASE_STATUS']] = df1_train_set[
    ['FULL_TIME_POSITION', 'PREVAILING_WAGE_RANGE', 'CASE_SUBMITTED_YEAR_RANGE', 'UNIV_EMPLOYER', 'OCCUPATION', 'WORKSITE_STATE',
     'CASE_STATUS']].apply(lambda x: x.astype('category'))

    #print(df1_train_set.head())

    X = df1_train_set.loc[:, 'FULL_TIME_POSITION':'WORKSITE_STATE']
    Y = df1_train_set.CASE_STATUS

    seed = 5
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.3, random_state=seed)
    #print(X_train.isnull().sum())

    X_train_encode = pd.get_dummies(X_train)
    X_val_encode = pd.get_dummies(X_validation)

    train_X = X_train_encode.values
    train_y = Y_train.values

    val_x = X_val_encode.values
    val_y = Y_validation.values
    print("Pre Processing is completed for training set")
    return train_X,train_y,val_x,val_y

def preprocessingTestingdata(user_input):
    print("Pre Processing data for test set")
    if(user_input=='1' or user_input=='3' or user_input=='2'):
        df_test = pd.read_csv('File 2 - H1B Dataset.csv',encoding="ISO-8859-1")
    
    merge_labels(df_test)
    fill_nan_values(df_test)
    prevailing_wage(df_test)
    case_submission_year_range(df_test)
    classify_employer(df_test)
    df1_test_set = df_test[
    ['FULL_TIME_POSITION', 'PREVAILING_WAGE_RANGE', 'CASE_SUBMITTED_YEAR_RANGE', 'UNIV_EMPLOYER', 'OCCUPATION', 'WORKSITE_STATE',
     'CASE_STATUS']].copy()

    df1_test_set[['FULL_TIME_POSITION', 'PREVAILING_WAGE_RANGE', 'CASE_SUBMITTED_YEAR_RANGE', 'UNIV_EMPLOYER', 'OCCUPATION', 'WORKSITE_STATE',
     'CASE_STATUS']] = df1_test_set[
    ['FULL_TIME_POSITION', 'PREVAILING_WAGE_RANGE', 'CASE_SUBMITTED_YEAR_RANGE', 'UNIV_EMPLOYER', 'OCCUPATION', 'WORKSITE_STATE',
     'CASE_STATUS']].apply(lambda x: x.astype('category'))
    class_mapping = {'CERTIFIED': 0, 'DENIED': 1}
    df1_test_set["CASE_STATUS"] = df1_test_set["CASE_STATUS"].map(class_mapping)

    X_test = df1_test_set.loc[:, 'FULL_TIME_POSITION':'WORKSITE_STATE']
    Y_test = df1_test_set.CASE_STATUS

    X_test_encode = pd.get_dummies(X_test)
    testX = X_test_encode.values
	
    testY = Y_test.values
    print("Pre Processing is completed for test set")
    return testX, testY