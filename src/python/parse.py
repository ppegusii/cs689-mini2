from __future__ import print_function
from collections import defaultdict
import numpy as np
import pandas as pd
import re
from sklearn import preprocessing as pp
import sys


# Returns training and testing data.
def trainTestData(fileNames, beginTestIdx=11, featureCnt=25):
    train, test = separateTrainTest(fileNames, beginTestIdx)
    trainXY = list(matrixXY(train, featureCnt))
    # print('trainXY[0].shape = {:s}'.format(trainXY[0].shape))
    # print('trainXY[1].shape = {:s}'.format(trainXY[1].shape))
    testXY = list(matrixXY(test, featureCnt))
    # print('trainXY = {:s}'.format(trainXY))
    trainXY[0], testXY[0] = normalize(trainXY[0], testXY[0])
    # print('trainXY = {:s}'.format(trainXY))
    # randomize
    ranIdx = np.random.permutation(trainXY[0].shape[0])
    trainXY[0] = trainXY[0][ranIdx]
    trainXY[1] = trainXY[1][ranIdx]
    return tuple(trainXY), tuple(testXY)


def normalize(train, test):
    scaler = pp.StandardScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    return train, test


# Convert the files to:
# X = None size=[n_samples,n_features]
# Y = None size=[n_samples]
# featureCnt = n_features
def matrixXY(defDictFiles, featureCnt):
    X = None  # size=[n_samples,n_features]
    Y = None  # size=[n_samples]
    for person, personFiles in defDictFiles.items():
        for personFile in personFiles:
            with open(personFile, 'rb') as f:
                df = pd.read_table(f, sep='   ', header=None, engine='python')
            df = df.iloc[:, 0:2]  # only keep height and width
            # Resample before extracting features.
            df = resample(df, featureCnt)
            if df is None:
                continue
            y = np.array([[classLabel(person)]])
            Y = (y if Y is None else
                 np.append(Y, [[classLabel(person)]], axis=0))
            x = extractFeatures(df)
            X = ([x] if X is None else
                 np.append(X, [x], axis=0))
    return X, Y


# Extract a size=[1,n_features] vector from the dataframe
# containing [[height,width],n_samples]
def extractFeatures(df):
    v = df.values
    # return v[:, 0]**2 / v[:, 1]  # best for KNN
    # return v[:, 0] / v[:, 1]**2
    # return v[:, 0]**-1.0 * v[:, 1]**-1.0
    # return v[:, 0] * v[:, 1]  # best for SVM
    # return v[:, 0]**2  * v[:, 1]  # best for SVM


# If I were to resample assuming even robot sampling rate, I could:
# tIndex = pd.date_range('1/1/1970', periods=16, freq='S')
# ts = pd.Series(df.iloc[:,0].values,index=tIndex)
# But if I reample so that all vectors are the same length,
# then I throw away knowledge of velocity.
# I could truncate it to 15.
# Losing 7 samples with less than 15, while some samples have
# 25 entries.
def resample(df, length=15):
    if length < 10 or length > 25:
        print('Given length outside of the length of data [10,25]'.format(
            length))
        sys.exit()
    # forward fill last value
    fill = np.empty((25 - df.shape[0], df.shape[1],))
    fill[:] = np.nan
    df = df.append(pd.DataFrame(fill), ignore_index=True)
    return df.fillna(method='ffill')
    # truncating
    if df.shape[0] < length:
        return None
    return df.iloc[0:length, :]


def classLabel(person):
    # using ascii code
    # return ord(person)
    return person


def separateTrainTest(fns, beginTestIdx):
    pattern = r'([a-g])([0-9]{1,2})\.dat'
    train = defaultdict(list)
    test = defaultdict(list)
    for fn in fns:
        match = re.search(pattern, fn)
        if not match:
            continue
        if int(match.group(2)) < beginTestIdx:
            train[match.group(1)].append(fn)
        else:
            test[match.group(1)].append(fn)
    return train, test
