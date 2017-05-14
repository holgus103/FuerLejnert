import numpy as np
from preprocessing import getData
from sklearn.svm import SVC 

test = ['test3.csv', 'test33.csv', 'test4.csv', 'test5.csv']
train = ['train3.csv', 'train33.csv', 'train4.csv', 'train5.csv']
sciezka = './dane/'

x_train, y_train = getData(sciezka + train[0])
x_test, y_test = getData(sciezka + test[0])



kategorie = np.unique(y_train)

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

    return score / y_test.size * 100

def MOC(C, kernel, tol, x_train, x_test, y_train, y_test):
    return compute(C, kernel, tol, x_train, x_test, y_train, y_test, [getValueForFirstSet, getValueForSecondSet])

print("acc: ", MOC(10, "rbf", 0.0001, x_train, x_test, y_train, y_test))

