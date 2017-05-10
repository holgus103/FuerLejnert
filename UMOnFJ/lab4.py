import numpy as np
from preprocessing import getData
from sklearn.svm import SVC 

test = ['test3.csv', 'test33.csv', 'test4.csv', 'test5.csv']
train = ['train3.csv', 'train33.csv', 'train4.csv', 'train5.csv']
sciezka = './dane/'

x_train, y_train = getData(sciezka + train[0])
x_test, y_test = getData(sciezka + test[0])



kategorie = np.unique(y_train)



def MOC(C, kernel, tol, x_train, x_test, y_train, y_test):
    labels_1 = []
    labels_2 = []
    test1_labels = []
    test2_labels = []
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
    for i in range(0,y_train.size):
        labels_1.insert(i, getValueForFirstSet(y_train[i]));
        labels_2.insert(i, getValueForSecondSet(y_train[i]));

    for i in range(0,y_test.size):
        test1_labels.insert(i, getValueForFirstSet(y_test[i]));
        test2_labels.insert(i, getValueForSecondSet(y_test[i]));

    bit1 = SVC(C=C, kernel=kernel, tol=tol).fit(x_train, labels_1)
    bit2 = SVC(C=C, kernel=kernel, tol=tol).fit(x_train, labels_2)

    score = 0
    pred1 = bit1.predict(x_test);
    pred2 = bit2.predict(x_test);

    for i in range(0, len(pred1)-1):
        if(pred1[i] == test1_labels[i] and pred2[i] == test2_labels[i]):
            score = score + 1

    return score / y_test.size * 100



print("acc: ", MOC(10, "rbf", 0.0001, x_train, x_test, y_train, y_test));

