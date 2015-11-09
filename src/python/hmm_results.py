from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re


def allAccuracy(dirName):
    acc = pd.DataFrame([accuracy(dn) for dn in
            [os.path.join(dirName, dn) for dn in next(os.walk(dirName))[1]]])
    acc = acc.astype(float)  # gets rid of key error
    acc.plot(kind='scatter', x='hidden states', y='mixture components',
             c='accuracy', s=50)
    plt.show()
    return acc


def accuracy(dirName):
    pattern = r'hs([0-9]{1,2})_mc([0-9]{1,2})'
    match = re.search(pattern, dirName)
    hs = match.group(1)
    mc = match.group(2)
    totalCorr = 0
    sampleCnt = 0
    for fn in [os.path.join(dirName, fn) for fn in next(os.walk(dirName))[2]]:
        df = read(fn)
        pattern = r'([a-g])\.dat'
        match = re.search(pattern, fn)
        person = match.group(1)
        totalCorr += corrCnt(df, person)
        sampleCnt += df.shape[1]
    ac = float(totalCorr)/sampleCnt if sampleCnt > 0 else np.nan
    return {
        'hidden states': hs,
        'mixture components': mc,
        'accuracy': ac,
        'ct': sampleCnt,
    }


def read(fileName):
    with open(fileName, 'rb') as f:
        df = pd.read_table(f, sep='  ', header=None, engine='python')
    df = pd.DataFrame(df.values, index=list('abcdefg'))
    df.columns = range(11, 16, 1)
    return df


def corrCnt(df, truth):
    pred = df.idxmax(axis=0)
    return len(pred[pred.str.contains(truth)])
    print(float(corrCnt)/len(pred))


def heatMap(df):
        # df.plot(kind='scatter')
        # plt.show()
        fig, ax = plt.subplots()
        heatmap = ax.pcolor(df, cmap=plt.cm.Blues)
        plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
        plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
        plt.colorbar(heatmap)
        plt.show()
