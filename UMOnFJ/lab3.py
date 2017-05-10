import numpy as np
from preprocessing import getData
from sklearn.svm import SVC 
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
test = ['test3.csv', 'test33.csv', 'test4.csv', 'test5.csv']
train = ['train3.csv', 'train33.csv', 'train4.csv', 'train5.csv']
sciezka = './dane/'
f = open('results.csv', 'w')
print('Uzywam sciezki: ' + sciezka)
params = [5,10,5, 10]
tols = [0.001, 0.0001, 0.0001, 0.0001]


def rbf_kernel(X,Y,gamma):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    gram_matrix[i, j] = np.exp(-gamma * np.sum((np.absolute(x - y)) ** 2))
        return gram_matrix

def laplacean(X,Y,gamma):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    gram_matrix[i, j] = np.exp(-gamma * np.sum((np.absolute(x - y))))
        return gram_matrix

def sinc(X,Y):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    sum = np.sum(np.absolute(x - y))
                    gram_matrix[i, j] = np.sinc(sum)
        return gram_matrix

def sinc2(X,Y):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    sum = np.sum(np.absolute(x - y)) ** 2
                    gram_matrix[i, j] = np.sinc(sum)
        return gram_matrix

def quadratic(X,Y,c):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    sum = np.sum(np.absolute(x - y)) ** 2
                    gram_matrix[i, j] = 1 - sum / (sum + c)
        return gram_matrix

def multiquadric(X,Y,c):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    sum = np.sum(np.absolute(x - y)) ** 2
                    gram_matrix[i, j] = - np.sqrt(sum + c * c)
        return gram_matrix

def inverse_multiquadric(X,Y,c):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    sum = np.sum(np.absolute(x - y)) ** 2
                    gram_matrix[i, j] = 1 / np.sqrt(sum + c * c)
        return gram_matrix

kernels = ['laplacean', 'sinc', 'quadratic', 'multiquadric']
cores = ['rbf', 'linear', 'poly', 'sigmoid'] 

def get_kernel(i, x, y, param):
    return {
    0: lambda x,y,param: laplacean(x,y,param),
    1: lambda x,y,param: sinc(x,y),
    2: lambda x,y,param: quadratic(x,y,param),
    3: lambda x,y,param: multiquadric(x,y,param),
    }[i](x,y,param)

for i in range(0,4):
    f.write("Zbior " + test[i] + ' ' + train[i] + '\n')
    f.write("kernel; c/gamma; tol; jakosc zbioru trenujacego (1vs1); jakosc zbiory testowego (1vs1); jakosc zbioru trenujacego (1vsR); jakosc zbiory testowego (1vsR)\n")
    X_train, y_train = getData(sciezka + train[i])
    X_test, y_test = getData(sciezka + test[i])
    for j in range(0,4):
        gram=get_kernel(j,X_train,X_train,10)
        gram_test=rbf_kernel(X_test,X_train,10)
        gram_test=rbf_kernel(X_test,X_train,10)
        print('Test gram calculated')
        #print('Jakosc klasyfikacji zbioru testowego: ', clf.score(gram_test,y_test))
        #cpredicted = clf.predict(gram_test)
        ovo = OneVsOneClassifier(SVC(C=params[j], kernel='precomputed', tol=params[j])).fit(gram, y_train)
        ovr = OneVsRestClassifier(SVC(C=params[j], kernel='precomputed', tol=params[j])).fit(gram, y_train)
        print('Jakosc klasyfikacji zbioru trenujacego: ', ovo.score(X_train, y_train))
        print('Jakosc klasyfikacji zbioru testowego: ', ovo.score(X_test,y_test))
        f.write(kernels[j])
        f.write(';')
        f.write(str(params[j]))
        f.write(';')
        f.write(str(tols[j]))
        gram_test = rbf_kernel(X_test,X_train,10)
        f.write(';')
        f.write(str(ovo.score(gram,y_train)))
        f.write(';')
        f.write(str(ovo.score(gram_test,y_test)))
        f.write(';')
        f.write(str(ovr.score(gram,y_train)))
        f.write(';')
        f.write(str(ovr.score(gram_test,y_test)))
        f.write('\n')
    for j in range(0,4):
        ovo = OneVsOneClassifier(SVC(C=params[j], kernel=cores[j], tol=tols[j])).fit(X_train, y_train)
        ovr = OneVsRestClassifier(SVC(C=params[j], kernel=cores[j], tol=tols[j])).fit(X_train, y_train)
        print('Jakosc klasyfikacji zbioru trenujacego: ', ovo.score(X_train, y_train))
        f.write(kernels[j])
        f.write(';')
        f.write(str(params[j]))
        f.write(';')
        f.write(str(tols[j]))
        f.write(';')
        f.write(str(ovo.score(X_train,y_train)))
        f.write(';')
        f.write(str(ovo.score(X_test,y_test)))
        f.write(';')

        f.write(str(ovr.score(X_train,y_train)))
        f.write(';')
        f.write(str(ovr.score(X_test,y_test)))
        f.write('\n')
f.close()