# -*- coding: utf-8 -*-
# -*- created_time: 2018/12/12 -*-

# from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
import random
from sklearn.preprocessing import LabelEncoder

# feature selection
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
# from sklearn.linear_model import LogisticRegression
# from sklearn.utils.extmath import log_logistic

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

from matplotlib import pyplot as plt
import time

import math
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
import warnings
warnings.filterwarnings("ignore")
# from imblearn.under_sampling import RandomUnderSampler

def load_mushroom_data(filepath = "mushroom.csv"):
    csv_data = pd.read_csv(filepath,header = None) # There are 22 dimensions (features) and1 target variable in the data

    data = csv_data.iloc[:,0:-1]
    target = csv_data.iloc[:,-1:]

    # convert Feature columns to numpy array
    data = np.array(data,dtype = "str")

    # convert target column to numpy array
    target = np.array(target, dtype = "str").reshape(len(target))
    # print(len(target))
    return data,target

def missing_data_process(data):
    # Get the 11th column(slices) with missing values in the dataset
    column_missing_data = data[:,10]
    column_missing_counter = Counter(column_missing_data)
    # Delete statistical results of missing values "?" in the dictionary of statistical results in the missing value column
    del column_missing_counter["?"]

    # The dictionary is sorted in ascending order according to the value of the element statistics result dictionary in
    # the missing value column (remove the missing value element "?"), and the result of the arrangement is a list of
    # each key-value pair tuple.
    # convert list column_missing_counter back to dictionary
    column_missing_counter = sorted(column_missing_counter.items(),key=lambda item:item[1])
    column_missing_counter = dict(column_missing_counter)

    # Weight-based random sampling
    # Based on element weights in missing value column, using the result of Weight-based random sampling
    # to replace missing values in missing value column
    def random_select_by_weight(column_missing_counter = column_missing_counter):
        total_weights = sum(column_missing_counter.values())
        rd = random.uniform(0,total_weights)
        curr_weight = 0
        value = None

        keys = column_missing_counter.keys()
        for key in keys:
            curr_weight += column_missing_counter[key]
            if rd <= curr_weight:
                value = key
                break
        return value

    row_num = data.shape[0] # row numbers
    # replace all missing data in 11th column
    for row in range(row_num):
        if data[row,10] == "?":
            data[row,10] = random_select_by_weight()

    # after_missing_process_counter = Counter(data[:,10])
    return data

def One_hot_LabelEncoder(data,target,data_one_hot_codong=True,n_components=10):
    if data_one_hot_codong:
        # using get_dummies function in pandas package to convert all features into one-hot coding
        # many dummy features would be produced
        data = pd.get_dummies(pd.DataFrame(data)) # 116 dummy features
        shape = data.shape[1]
        # using PCA to do the dimensionality reduction for 116 dummy features
        pca = PCA(n_components=n_components)
        pca.fit(data)
        data = pca.transform(data)
        data = np.array(data, dtype="float")
    else:
        # Via LabelEncoder class in sklearn.preprocessing, encode all features (Label Encoder)
        # Next step should be feature selection
        col_num = data.shape[1]
        for col in range(col_num):
            data[:,col] = LabelEncoder().fit_transform(data[:,col])
        data = np.array(data, dtype="float")

    # Via LabelEncoder class in sklearn.preprocessing, encode target variable (Label Encoder)
    target = LabelEncoder().fit_transform(target)
    target = np.array(target, dtype="float")

    return data,target

def data_feature_selection(data,target):

    # SelectKBest(mutual_info_classif, k=10) select features through mutual_info_classif
    # select top 10 important features
    fea_selector = SelectKBest(mutual_info_classif, k=10)
    fea_selector.fit_transform(data, target)

    data_new = None
    for i in range(len(fea_selector.get_support())):
        if data_new is None:
            if fea_selector.get_support()[i] == True:
                data_new = data[:,i]
        else:
            if fea_selector.get_support()[i] == True:
                data_new = np.column_stack((data_new, data[:,i]))

    # Using the decision tree-based feature selection method to estimate the importance of the 10 most important
    # features selected in the previous step
    treeClf = ExtraTreesClassifier(n_estimators=100)
    treeClf.fit(data_new,target)
    feature_importances = list(treeClf.feature_importances_)
    feature_num = list(i for i in range(10))

    # plt.figure(figsize=(18,12))
    # plt.xticks(np.arange(len(feature_num)), feature_num,fontsize =13)
    # plt.bar(feature_num,feature_importances,color="darkcyan")
    # plt.xlabel("Feature Index",fontsize=15)
    # plt.ylabel("Feature Importance",fontsize=15)
    # plt.title("The Importances of Different Features",fontsize=17)
    # plt.savefig("feature_importances.png")
    # plt.show()

    return data_new,target

def load_survival_csv_data(filepath = "survival.csv"):
    csv_data = pd.read_csv(filepath,header = None)

    # Feature columns
    data = csv_data.iloc[:,0:-1]
    # target column
    target = csv_data.iloc[:,-1:]

    # convert Feature columns to numpy array
    data = np.array(data,dtype = "float")
    # convert target column to numpy array
    target = np.array(target, dtype = "float").reshape(len(target))
    # print(len(target))
    return data,target

# standardization
def standard(data):
    standardData = data.copy()

    rows = standardData.shape[0]
    cols = standardData.shape[1]

    for col_num in range(cols):
        sd = np.std(standardData[:,col_num])
        mean = np.mean(standardData[:,col_num])

        for row_num in range(rows):
            standardData[row_num,col_num] = (standardData[row_num, col_num] - mean)/sd

    return standardData

# programme Logistic Regression from scratch
# Note!!!:
# If tolerance is slow to converge near the threshold,
# then you need to restart the logistic regression algorithm to reinitialize the weights and bias.
class Logistic_Regression():
    def __init__(self,data=None,target=None,learning_rate = 0.01,threshold = 0.0001,max_iteration = 1000,scoring = "accuracy"):
        self._data = data
        self._target = target
        self._weights = None
        self._bias = None
        self._learning_rate = learning_rate
        self._threshold = threshold
        self._max_iteration = max_iteration
        self._scoring = scoring  # scoring parameter can be：/accuracy/precision/recall/f1/roc_auc_score

    # Initialize weights
    def _init_weights(self):
        num_coefficients = self._data.shape[1]
        # Return a sample (or samples) from the “standard normal” distribution.
        # self._weights = np.random.randn(num_coefficients,1)/ num_coefficients
        self._weights = np.zeros((num_coefficients, 1))
        return self._weights

    # Initialize bias
    def _init_bias(self):
        # Return a sample (or samples) from the “standard normal” distribution.
        self._bias = np.random.randn(1)
        return self._bias

    # Order items randomly
    def _order_randomly(self):
        self._target = self._target.reshape(len(self._target),1)
        all_data = np.hstack((self._data,self._target))
        np.random.shuffle(all_data)
        self._data = all_data[:,:-1]
        self._target = all_data[:,-1:]
        return self._data, self._target

    # train weights and bias
    def fit(self,data=None,target=None):
        if self._data is None and self._target is None:
            self._data = data
            self._target = target

        time_start = time.time()

        # Initialize weights and bias
        self._weights = self._init_weights()
        self._bias = self._init_bias()

        # Initialize loss and iteration time
        loss = float("inf")
        previous_loss = 0
        iteration = 0
        # the predicted class labels
        predicts = None
        # the variable "active" represents the status of the while loop
        active = True

        while active:
            # Order items randomly
            self._data, self._target = self._order_randomly()

            # result of logistic regression
            predicts = 1 / (1 + np.exp(-(np.dot(self._data, self._weights) + self._bias)))

            # Use the absolute value loss divided by the number of training data as a loss function
            # loss = np.sum(np.abs(self._target - predicts),axis=0)[0] / (self._target.shape[0])
            # Calculate the loss using the formula of  loss function of the logistic regression in the literature,
            # When tol is greater than threshold,then keep doing stochastic gradient decsent
            loss = np.mean(-self._target * np.log(predicts) - (1-self._target) * np.log(1 - predicts),axis=0)
            tol = previous_loss - loss
            previous_loss = loss

            for row in range(self._data.shape[0]):
                self._bias = self._bias - self._learning_rate * (predicts[row,0] - self._target[row,0]) * predicts[row,0] * (1-predicts[row,0]) * 1

                for col in range(self._data.shape[1]):
                    self._weights[col,0] = self._weights[col,0] - self._learning_rate * (predicts[row,0] - self._target[row,0]) \
                                           * predicts[row,0] * (1-predicts[row,0]) * self._data[row,col]

            iteration += 1
            # if statement controls the status of the variable "active"
            if (tol <= self._threshold and tol > 0) or (iteration >= self._max_iteration):
                active = False
            # print("Loss in each epoch is: ", loss)

        # round predicts
        predicts = np.round(predicts)

        print("\nWeights are: ",self._weights)
        print("\nBias is: ", self._bias)
        print("\nFinal Loss is: ",loss)
        print("Iteration times: ",iteration)

        # calculate accuracy, precision, recall, f1, roc_auc
        if self._scoring == None:
            print("")
        if self._scoring == "accuracy":
            accuracy = accuracy_score(self._target,predicts)
            print("\nAccuracy: ",accuracy)
        if self._scoring == "precision":
            precision = precision_score(self._target,predicts)
            print("\nPrecision: ", precision)
        if self._scoring == "recall":
            recall = recall_score(self._target, predicts)
            print("\nRecall: ", recall)
        if self._scoring == "f1":
            f1= f1_score(self._target, predicts)
            print("\nF1: ", f1)
        if self._scoring == "roc_auc":
            roc_auc = roc_auc_score(self._target, predicts)
            print("\nRoc_Auc: ", roc_auc)

        time_end = time.time()
        print("Time cost(training model): ",str(time_end - time_start),"seconds\n")

        return self._weights,self._bias

    def predict(self, data):
        predicts = 1 / (1 + np.exp(-(np.dot(data, self._weights) + self._bias)))
        predicts = np.round(predicts)

        return predicts

    # Using training test to train the weights and the bias. Then, using test set to test
    # the accuracy/precision/recall/f1 of the Logistic Regression model
    def train_test_predict(self,data,target,train_size = 0.6):
        time_start = time.time()
        data_train, data_test, target_train, target_test = train_test_split(data, target, train_size=train_size)
        self.fit(data_train, target_train)
        test_predicts = self.predict(data_test)

        accuracy = accuracy_score(target_test, test_predicts)
        print("\nAccuracy(Test set): ", accuracy)

        precision = precision_score(target_test, test_predicts)
        print("Precision(Test set): ", precision)

        recall = recall_score(target_test, test_predicts)
        print("Recall(Test set): ", recall)

        f1 = f1_score(target_test, test_predicts)
        print("F1(Test set): ", f1)

        roc_auc = roc_auc_score(target_test, test_predicts)
        print("Roc_Auc(Test set): ", roc_auc)

        time_end = time.time()
        print("Time cost(Train Test Split): ", str(time_end - time_start), "seconds")

    # doing cross-validation for Logistic Regression
    def corss_val_score(self,data,target,cv=5,scoring_plot = None):
        time_start = time.time()
        test_size = 1/cv
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        rocs = []

        for i in range(cv):
            data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=test_size)
            self.fit(data_train, target_train)
            test_predicts = self.predict(data_test)

            accuracies.append(accuracy_score(target_test, test_predicts))
            precisions.append(precision_score(target_test, test_predicts))
            recalls.append(recall_score(target_test, test_predicts))
            f1s.append(f1_score(target_test, test_predicts))
            rocs.append(roc_auc_score(target_test, test_predicts))

        print("\nAccuracy(" + str(cv) + " cross-validations): ", np.mean(accuracies))
        print("Precision(" + str(cv) + " cross-validations): ", np.mean(precisions))
        print("Recall(" + str(cv) + " cross-validations): ", np.mean(recalls))
        print("F1(" + str(cv) + " cross-validations): ", np.mean(f1s))
        print("Roc_Auc(" + str(cv) + " cross-validations): ", np.mean(rocs))

        time_end = time.time()
        print("\nTime cost(" + str(cv) + " cross-validations): ", str(time_end - time_start), "seconds")
        # Time cost(5 cross-validations):  15.475885391235352 seconds


        if scoring_plot is not None:
            corss_validation_times = [i+1 for i in range(cv)]

            plt.figure(figsize=(12, 8))
            plt.xticks(corss_validation_times)
            plt.xlabel("Corss-validation times", fontsize=15)
            plt.ylabel(scoring_plot.title(), fontsize=15)

            if scoring_plot == "accuracy":
                plt.plot(corss_validation_times, accuracies, color="darkcyan", marker="v")
                plt.title("The " + scoring_plot.title() + " of Cross-validaion of LogisticRegression" +
                          "(Mean " + scoring_plot.title() + ": " + str(np.mean(accuracies)) + ")",fontsize=17)
            if scoring_plot == "precision":
                plt.plot(corss_validation_times, precisions, color="darkcyan", marker="v")
                plt.title("The " + scoring_plot.title() + " of Cross-validaion of LogisticRegression" +
                          "(Mean " + scoring_plot.title() + ": " + str(np.mean(precisions)) + ")",fontsize=17)
            if scoring_plot == "recall":
                plt.plot(corss_validation_times, recalls, color="darkcyan", marker="v")
                plt.title("The " + scoring_plot.title() + " of Cross-validaion of LogisticRegression" +
                          "(Mean " + scoring_plot.title() + ": " + str(np.mean(recalls)) + ")",fontsize=17)
            if scoring_plot == "f1":
                plt.plot(corss_validation_times, f1s, color="darkcyan", marker="v")
                plt.title("The " + scoring_plot.title() + " of Cross-validaion of LogisticRegression" +
                          "(Mean " + scoring_plot.title() + ": " + str(np.mean(f1s)) + ")",fontsize=17)
            if scoring_plot == "roc_auc":
                plt.plot(corss_validation_times, rocs, color="darkcyan", marker="v")
                plt.title("The " + scoring_plot.title() + " of Cross-validaion of LogisticRegression" +
                          "(Mean " + scoring_plot.title() + ": " + str(np.mean(rocs)) + ")",fontsize=17)

            plt.savefig("corss_validation_" + scoring_plot.title() + "_LogisticRegression.png")
            plt.show()
            plt.close()

from scipy import stats

class Logistic_Bagging():
    def __init__(self,data = None, target = None,estimators = 10,learning_rate = 0.01,threshold = 0.0001,max_iteration = 800,scoring = "accuracy"):
        self._data = data
        self._target = target.reshape(target.shape[0],1) # Change target variable from one-dimensional array to two-dimensional array
        self._trainingSetData = None
        self._trainingSetTarget = None
        self._testSetData = None
        self._testSetTarget = None

        self._weights = None
        self._bias = None

        self._estimators = estimators
        self._learning_rate = learning_rate
        self._threshold = threshold
        self._max_iteration = max_iteration
        self._scoring = scoring  # scoring parameter can be：/accuracy/precision/recall/f1/roc_auc_score
        self.lr = Logistic_Regression(learning_rate = self._learning_rate,threshold = self._threshold,max_iteration = self._max_iteration,scoring = self._scoring)

    # split original data to training set and test set
    def train_test_split(self,data=None,target=None,trainingSet_proportion = 0.7):
        if self._data is None and self._target is None:
            self._data = data
            self._target = target
        trainingSize =  math.ceil(self._target.shape[0] * trainingSet_proportion)
        trainingSetGroup = np.random.choice(np.arange(0, self._target.shape[0]), size=trainingSize, replace=False)

        # Can slice the numpy array by placing the list of test set row index numbers directly at the row index of numpy array.
        trainingSetData = self._data[trainingSetGroup,:]
        # If the target variable is an one dimensional array, then only  needs  one dimensional index in slice.
        trainingSetTarget = self._target[trainingSetGroup,:]
        # Delete the training set row index number from the original data set, and the rest is the test set.
        testSetData = np.delete(self._data, trainingSetGroup, axis=0)
        testSetTarget = np.delete(self._target, trainingSetGroup, axis=0)

        self._trainingSetData = trainingSetData
        self._trainingSetTarget = trainingSetTarget
        self._testSetData = testSetData
        self._testSetTarget = testSetTarget

        # return trainingSetData, trainingSetTarget, testSetData, testSetTarget


    # Take bootstrap samples from the original population and hand class imbalance
    def bootstrap_sampling(self, training_samples_proportion = 1):

        # Select the training set data  by randomly selecting the row index numbers from the original data set.
        training_samples = math.ceil(self._trainingSetTarget.shape[0] * training_samples_proportion)
        # bootstrap sampling (with replacement in each sampling)
        trainingGroup = np.random.choice(np.arange(0,self._trainingSetTarget.shape[0]), size=training_samples, replace = True)
        trainingData = self._trainingSetData[trainingGroup,:]
        trainingTarget = self._trainingSetTarget[trainingGroup,:]
        # Convert the two-dimensional array trainingTarget into a one-dimensional array for convenient  processing by the latter smote algorithm
        trainingTarget = trainingTarget.reshape(trainingTarget.shape[0])

        # Perform category statistics on the target variable trainingTarget (use the Counter for category statistics for this column)
        class_counter = Counter(trainingTarget)
        # The dictionary is sorted in ascending order according to the value of the target variable statistical result
        # dictionary, and the result after the arrangement is a list of each key element pair of tuples.,
        # Then use the dict function to convert the sorted list of tuples of key-values back to the dictionary, which is
        # to facilitate the processing of the imbalanced class.
        class_counter = sorted(class_counter.items(), key=lambda item: item[1])
        class_counter = dict(class_counter)

        # handing class imbalance
        # define SMOTE model，ratio defines the number of one class after somte resampling.
        smote = SMOTE(random_state=33, ratio={list(class_counter.keys())[0] : list(class_counter.values())[1]})
        trainingData,trainingTarget = smote.fit_resample(trainingData,trainingTarget)
        trainingTarget = trainingTarget.reshape(trainingTarget.shape[0],1)  # Convert a one-dimensional array trainingTarget to a two-dimensional array

        # class_counter_after_imbalance = Counter(trainingTarget[:,0])

        return trainingData, trainingTarget

    def fit(self,trainingSet_proportion = 0.7):
        self.train_test_split(trainingSet_proportion = trainingSet_proportion)

        self._weights = []
        self._bias = []
        time_start = time.time()

        # Bootstrap sampling from the overall data, training T weak classifiers using bootstrap sampling
        for i in range(self._estimators):
            trainingData,trainingTarget = self.bootstrap_sampling()

            lr = self.lr
            weights,bias = lr.fit(trainingData,trainingTarget)
            self._weights.append(weights)
            self._bias.append(bias)

        time_end = time.time()
        print("Time cost of training bagging(from scratch): ", str(time_end - time_start), "seconds")


    def predict(self,testData = None,testTarget = None,scoring = "accuracy",output = True):
        predicts_sets = []
        predict_accuracies = []

        if testData is None and testTarget is None:
            # copy original data "self._data", and assign it to the variable "data"
            testData = self._testSetData.copy()
            testTarget = self._testSetTarget.copy()

        # presict labels of test data by each weak classifier's weight and bias
        for i in range(len(self._weights)):
            predicts = 1 / (1 + np.exp(-(np.dot(testData, self._weights[i]) + self._bias[i])))
            # round predicts
            predicts = np.round(predicts)
            predicts_sets.append(predicts)
            predict_accuracies.append(accuracy_score(testTarget,predicts))

        # The list predicts_sets of all weak classifier prediction label results will be stored, and all the array
        # elements in the list will be merged into an array with the same number of rows as the testData and the same
        # number of columns as the weak classifier with the hstack method.
        predicts_sets_integrate = np.hstack((predicts_sets[i] for i in range(len(predicts_sets))))

        # class labels after simple majority voting (the mode in each row along the column axis)
        predicts_sets_modes = stats.mode(predicts_sets_integrate,axis=1)[0]
        # predicts_sets_modes = (np.median(predicts_sets,axis=0)).round()

        accuracy = None
        # if scoring is not None:
        #     if scoring == "accuracy":
        #         accuracy = accuracy_score(testTarget,predicts_sets_modes)
        #         print("\nAccuracy(Test set): ", accuracy)
        #     if scoring == "precision":
        #         precision = precision_score(testTarget,predicts_sets_modes)
        #         print("\nPrecision(Test set): ", precision)
        #     if scoring == "recall":
        #         recall = recall_score(testTarget,predicts_sets_modes)
        #         print("\nRecall(Test set): ", recall)
        #     if scoring == "f1":
        #         f1 = f1_score(testTarget,predicts_sets_modes)
        #         print("\nF1(Test set): ", f1)
        #     if scoring == "roc_auc":
        #         roc_auc = roc_auc_score(testTarget,predicts_sets_modes)
        #         print("Roc_Auc(Test set): ", roc_auc)

        print("\nBagging from scratch using test data")
        accuracy = accuracy_score(testTarget, predicts_sets_modes)
        print("Accuracy(Test set): ", accuracy)

        precision = precision_score(testTarget, predicts_sets_modes)
        print("Precision(Test set): ", precision)

        recall = recall_score(testTarget, predicts_sets_modes)
        print("Recall(Test set): ", recall)

        f1 = f1_score(testTarget, predicts_sets_modes)
        print("F1(Test set): ", f1)

        roc_auc = roc_auc_score(testTarget, predicts_sets_modes)
        print("Roc_Auc(Test set): ", roc_auc)

        # if output:
        #     output_data = np.hstack((self._testSetData, self._testSetTarget,predicts_sets_modes))
        #     df_output_data = pd.DataFrame(output_data)
        #     df_output_data.to_csv("tested_dataset",index=False)

        return accuracy
        # print("ok")



class skelearn_loggistic_bagging():
    def __init__(self,data = None, target = None,estimators = 20,threshold = 0.0001,max_iteration = 100,scoring = "accuracy"):
        self._data = data
        self._target = target.reshape(target.shape[0], 1)  # Change target variable from one-dimensional array to two-dimensional array
        self._trainingSetData = None
        self._trainingSetTarget = None
        self._testSetData = None
        self._testSetTarget = None

        # parameters of logistic regression
        self._threshold = threshold
        self._max_iteration = max_iteration
        # parameters of bagging
        self._estimators = estimators

        self._scoring = scoring  # scoring parameter can be：/accuracy/precision/recall/f1/roc_auc_score
        # regard logistic regression and bagging models as attributes
        self.lr = LogisticRegression(tol=self._threshold, max_iter=self._max_iteration, fit_intercept=True)
        self.bagging = BaggingClassifier(self.lr, n_estimators=self._estimators, max_samples=1.0, bootstrap=True)

    # split original data to training set and test set
    def train_test_split(self,data=None,target=None,trainingSet_proportion = 0.7):
        if self._data is None and self._target is None:
            self._data = data
            self._target = target
        trainingSize =  math.ceil(self._target.shape[0] * trainingSet_proportion)
        trainingSetGroup = np.random.choice(np.arange(0, self._target.shape[0]), size=trainingSize, replace=False)

        # Can slice the numpy array by placing the list of test set row index numbers directly at the row index of numpy array.
        trainingSetData = self._data[trainingSetGroup,:]
        # If the target variable is an one dimensional array, then only  needs  one dimensional index in slice.
        trainingSetTarget = self._target[trainingSetGroup,:]
        # Delete the training set row index numbers from the original data set, and the rest is the test set.
        testSetData = np.delete(self._data, trainingSetGroup, axis=0)
        testSetTarget = np.delete(self._target, trainingSetGroup, axis=0)

        self._trainingSetData = trainingSetData
        self._trainingSetTarget = trainingSetTarget
        self._testSetData = testSetData
        self._testSetTarget = testSetTarget

    def fit(self, trainingSet_proportion = 0.7):
        time_start = time.time()
        self.train_test_split(trainingSet_proportion = trainingSet_proportion)
        self.bagging.fit(self._trainingSetData,self._trainingSetTarget)

        time_end = time.time()
        print("\nTime cost of training bagging(sklearn): ", str(time_end - time_start), "seconds")

    def predict(self,testData = None,testTarget = None,scoring = "accuracy",output = True):
        if testData is None and testTarget is None:
            # copy original data "self._data", and assign it to the variable "data"
            testData = self._testSetData.copy()
            testTarget = self._testSetTarget.copy()

            predicts_sets_modes = self.bagging.predict(testData)

        accuracy = None
        # if scoring is not None:
        #     if scoring == "accuracy":
        #         accuracy = accuracy_score(testTarget,predicts_sets_modes)
        #         print("\nAccuracy(Test set): ", accuracy)
        #     if scoring == "precision":
        #         precision = precision_score(testTarget,predicts_sets_modes)
        #         print("\nPrecision(Test set): ", precision)
        #     if scoring == "recall":
        #         recall = recall_score(testTarget,predicts_sets_modes)
        #         print("\nRecall(Test set): ", recall)
        #     if scoring == "f1":
        #         f1 = f1_score(testTarget,predicts_sets_modes)
        #         print("\nF1(Test set): ", f1)
        #     if scoring == "roc_auc":
        #         roc_auc = roc_auc_score(testTarget,predicts_sets_modes)
        #         print("Roc_Auc(Test set): ", roc_auc)

        print("\nBagging of sklearn library using test data")
        accuracy = accuracy_score(testTarget,predicts_sets_modes)
        print("Accuracy(Test set): ", accuracy)

        precision = precision_score(testTarget,predicts_sets_modes)
        print("Precision(Test set): ", precision)

        recall = recall_score(testTarget,predicts_sets_modes)
        print("Recall(Test set): ", recall)

        f1 = f1_score(testTarget,predicts_sets_modes)
        print("F1(Test set): ", f1)

        roc_auc = roc_auc_score(testTarget,predicts_sets_modes)
        print("Roc_Auc(Test set): ", roc_auc)

        # if output:
        #     output_data = np.hstack((self._testSetData, self._testSetTarget, predicts_sets_modes))
        #     df_output_data = pd.DataFrame(output_data)
        #     df_output_data.to_csv("tested_dataset.csv", index=False)

        return accuracy
        # print("ok")

def accuracy_test_plot(data,target,Threshold = True, Learning_rates = True, Estimators = True):
    thresholds = np.linspace(0.0001, 0.01, 20)
    learning_rates = np.linspace(0.001,0.1,20)
    estimators = np.linspace(10,20,10,dtype=np.int32)
    accuracies_logistic_bagging = []
    accuracies_sklearn_bagging = []

    if Threshold:
        for threshold in thresholds:
            lb = Logistic_Bagging(data=data, target=target, estimators=10, learning_rate=0.01, threshold=threshold,max_iteration=800,scoring="accuracy")
            lb.fit(trainingSet_proportion=0.8)
            accuracies_logistic_bagging.append(lb.predict())

            skb = skelearn_loggistic_bagging(data=data, target=target, estimators=10, threshold=threshold,max_iteration=800,scoring="accuracy")
            skb.fit(trainingSet_proportion=0.8)
            accuracies_sklearn_bagging.append(skb.predict())

        plt1 = plt
        plt1.figure(figsize=(12, 8))
        plt1.plot(thresholds, accuracies_logistic_bagging, marker='*', color='chocolate',label='Bagging(logistic regression) from scratch')
        plt1.plot(thresholds, accuracies_sklearn_bagging, marker='v', color='darkcyan',label='Bagging(logistic regression) from sklearn')
        plt1.xlabel('Thresholds', fontsize=20)
        plt1.ylabel('Accuracy', fontsize=20)
        plt1.title("Accuracies with different thresholds", fontsize=27)
        plt1.legend(loc='best')
        plt1.savefig("accuracy_threshold_plot.png")
        # plt1.show()
        plt1.close()

    if Learning_rates:
        accuracies_logistic_bagging = []
        for learning_rate in learning_rates:
            lb = Logistic_Bagging(data=data, target=target, estimators=10, learning_rate=learning_rate, threshold=0.0001,max_iteration=50,scoring="accuracy")
            lb.fit(trainingSet_proportion=0.8)
            accuracies_logistic_bagging.append(lb.predict())

        plt1 = plt
        plt1.figure(figsize=(12, 8))
        plt1.plot(learning_rates, accuracies_logistic_bagging, marker='*', color='chocolate',label='Bagging(logistic regression) from scratch')
        plt1.xlabel('Learning rates', fontsize=20)
        plt1.ylabel('Accuracy', fontsize=20)
        plt1.title("Accuracies with different learning rates", fontsize=27)
        plt1.legend(loc='best')
        plt1.savefig("accuracy_Learning_rates_plot.png")
        # plt1.show()
        plt1.close()

    if Estimators:
        accuracies_logistic_bagging = []
        accuracies_sklearn_bagging = []
        for estimator in estimators:
            lb = Logistic_Bagging(data=data, target=target, estimators=estimator, learning_rate=0.01, threshold=0.0001,max_iteration=50,scoring="accuracy")
            lb.fit(trainingSet_proportion=0.8)
            accuracies_logistic_bagging.append(lb.predict())

            skb = skelearn_loggistic_bagging(data=data, target=target, estimators=estimator, threshold=0.0001,max_iteration=50, scoring="accuracy")
            skb.fit(trainingSet_proportion=0.8)
            accuracies_sklearn_bagging.append(skb.predict())

        plt1 = plt
        plt1.figure(figsize=(12, 8))
        plt1.plot(estimators, accuracies_logistic_bagging, marker='*', color='chocolate',label='Bagging(logistic regression) from scratch')
        plt1.plot(estimators, accuracies_sklearn_bagging, marker='v', color='darkcyan',label='Bagging(logistic regression) from sklearn')
        plt1.xlabel('Estimators', fontsize=20)
        plt1.ylabel('Accuracy', fontsize=20)
        plt1.title("Accuracies with different estimators", fontsize=27)
        plt1.legend(loc='best')
        plt1.savefig("accuracy_estimators_plot.png")
        # plt1.show()
        plt1.close()

if __name__ == "__main__":
    # test mushroom dataset
    data, target = load_mushroom_data()
    data = missing_data_process(data)
    data, target = One_hot_LabelEncoder(data, target, data_one_hot_codong=True, n_components=15)
    data, target = data_feature_selection(data, target)

    # lr = Logistic_Regression(learning_rate=0.01, threshold=0.0001, scoring="accuracy")
    # # lr.train_test_predict(data, target, train_size = 0.6)
    # lr.corss_val_score(data, target, cv=5, scoring_plot="accuracy")

    lb = Logistic_Bagging(data=data, target=target, estimators=14, learning_rate=0.01, threshold=0.0001,max_iteration = 800,scoring="accuracy")
    # lb.bootstrap_sampling()
    lb.fit(trainingSet_proportion=0.8)
    lb.predict()

    skb = skelearn_loggistic_bagging(data = data, target = target,estimators = 14,threshold = 0.0001,max_iteration = 800,scoring = "accuracy")
    skb.fit(trainingSet_proportion=0.8)
    skb.predict()



    # # test survival dataset
    # data,target = load_survival_csv_data()
    # data = standard(data)
    #
    # # lr = Logistic_Regression(learning_rate=0.01, threshold=0.0001, scoring="accuracy",max_iteration=50)
    # # # lr.train_test_predict(data, target, train_size = 0.6)
    # # lr.corss_val_score(data, target, cv=5, scoring_plot="accuracy")
    #
    # lb = Logistic_Bagging(data=data,target=target,estimators = 13,learning_rate = 0.01,threshold = 0.0001,max_iteration = 50,scoring="accuracy")
    # # lb.bootstrap_sampling()
    # lb.fit(trainingSet_proportion = 0.8)
    # lb.predict()
    #
    # skb = skelearn_loggistic_bagging(data=data, target=target, estimators=15, threshold=0.0001, max_iteration=50,scoring="accuracy")
    # skb.fit(trainingSet_proportion=0.8)
    # skb.predict()

    # accuracy_test_plot(data, target, Threshold=True, Learning_rates=True, Estimators=True)
