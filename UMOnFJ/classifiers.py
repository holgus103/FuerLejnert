
from sklearn.svm import SVC
import numpy as np

def compute(C, kernel, tol, x_train, x_test, y_train, y_test, funcs):
    train_labels = []
    test_labels = []
    count = len(funcs)
    # initialize lists
    for i in range(0, count):
        train_labels.insert(i,[])
        test_labels.insert(i,[])
    # initialize train label lists
    for i in range(0,y_train.size):
        for j in range(0, count):
            train_labels[j].insert(i, funcs[j](y_train[i]));

    for i in range(0,y_test.size):
        for j in range(0, count):
            test_labels[j].insert(i, funcs[j](y_test[i]));

    bits = [];
    pred = [];
    for i in range(0, count):
        bits.insert(i, SVC(C=C, kernel=kernel, tol=tol).fit(x_train, train_labels[i]))

    score = 0;
    for i in range(0, count):
        pred.insert(i, bits[i].predict(x_test));

    for i in range(0, pred[0].size):
        res = True
        for j in range(0, len(pred)):
            res = res and pred[j][i] == test_labels[j][i]
        if res :
          score = score + 1

    return float(score) / y_test.size * 100


def MOC4(C, kernel, tol, x_train, x_test, y_train, y_test):
    def getValueForFirstSet(val):
            if val > 2 :
                return 1
            else:
                return -1

    def getValueForSecondSet(val):
            if val % 2 == 0 :
                return 1
            else:
                return -1
    return compute(C, kernel, tol, x_train, x_test, y_train, y_test, [getValueForFirstSet, getValueForSecondSet])

def base(val, index):
    if val == index:
        return 1
    else:
        return -1


def OneVsAll4(C, kernel, tol, x_train, x_test, y_train, y_test):
    funcs = [lambda val: base(val, 1),lambda val: base(val, 2),lambda val: base(val, 3),lambda val: base(val, 4) ]
    return compute(C, kernel, tol, x_train, x_test, y_train, y_test, funcs)

def OneVsAll16(C, kernel, tol, x_train, x_test, y_train, y_test):
    funcs = []
    for i in range(1, 17):
        funcs.insert(i, lambda val: base(val, i))
    return compute(C, kernel, tol, x_train, x_test, y_train, y_test, funcs)

def mocBase(val, i):
    print "val: ", val
    print "i: ", i
    if (int(val) & (1 << (i-1))) > 0:
        return 1
    else:
        return -1

def MOC16(C, kernel, tol, x_train, x_test, y_train, y_test):
    funcs = []
    for i in range(1, 5):
        funcs.insert(i, lambda val:  mocBase(val, i))
    return compute(C, kernel, tol, x_train, x_test, y_train, y_test, funcs)
