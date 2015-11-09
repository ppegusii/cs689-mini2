from __future__ import print_function
import pandas as pd
from sklearn import svm, grid_search
from sklearn.neighbors import KNeighborsClassifier as knn


def SVM(trainXY, testXY):
    print('trainXY[0].shape = {:s}'.format(trainXY[0].shape))
    print('trainXY[1][:, 0].shape = {:s}'.format(trainXY[1][:, 0].shape))
    print('type(trainXY[0]) = {:s}'.format(type(trainXY[0])))
    print('type(trainXY[1])= {:s}'.format(type(trainXY[1])))
    # clf = svm.SVC(decision_function_shape='ovr')
    params = {
        'kernel': ('linear', 'rbf', 'sigmoid', 'poly'),
        'C': [0.25, 0.5, 1, 2, 3, 10, 11],
        'class_weight': ['balanced', None],
        'tol': [0.001, 0.0001, 0.00001],
    }
    svr = svm.SVC()
    clf = grid_search.GridSearchCV(svr, params)
    # clf = svm.SVC()
    # clf = svm.SVC(kernel='rbf')
    # clf = svm.NuSVC(nu=0.01, kernel='linear')
    print('clf = {:s}'.format(clf))
    clf.fit(trainXY[0], trainXY[1][:, 0])  # grid wants shape y = (n,)
    print('clf.best_estimator_ = {:s}'.format(clf.best_estimator_))
    clf = clf.best_estimator_
    # dec = clf.decision_function([[1]])
    prd = clf.predict(testXY[0])
    print('prd = {:s}'.format(prd))
    print('ans = {:s}'.format(testXY[1][:, 0].transpose()))
    print('accuracy = {:f}'.format(
        accuracy(prd, testXY[1][:, 0].transpose())))


def KNN(trainXY, testXY):
    # clf = knn(n_neighbors=3)
    params = {
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'weights': ['uniform', 'distance'],
    }
    clf = knn()
    clf = grid_search.GridSearchCV(clf, params)
    print('clf = {:s}'.format(clf))
    clf.fit(trainXY[0], trainXY[1][:, 0])
    print('clf.best_estimator_ = {:s}'.format(clf.best_estimator_))
    clf = clf.best_estimator_
    prd = clf.predict(testXY[0])
    print('prd = {:s}'.format(prd))
    print('ans = {:s}'.format(testXY[1][:, 0].transpose()))
    print('accuracy = {:f}'.format(
        accuracy(prd, testXY[1][:, 0].transpose())))


def accuracy(pred, ans):
    df = pd.DataFrame({'pred': pred, 'ans': ans})
    correct = df[df['pred'] == df['ans']]
    return float(correct.shape[0])/df.shape[0]
