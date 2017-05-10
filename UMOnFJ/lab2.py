import numpy as np
from preprocessing import getData
from sklearn.svm import SVC
test = ['test1.csv', 'test11.csv', 'test2.csv', 'test22.csv' ]
train = ['train1.csv', 'train11.csv', 'train2.csv', 'train22.csv' ]
sciezka = './'
f = open('results.csv', 'w')
print('Uzywam sciezki: ' + sciezka)
params = [5,10,0,0,0,5,10,5,10,10]


def rbf_kernel(X,Y,gamma):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    gram_matrix[i, j] = np.exp(-gamma*np.sum((np.absolute(x-y))**2))
        return gram_matrix

def laplacean(X,Y,gamma):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    gram_matrix[i, j] = np.exp(-gamma*np.sum((np.absolute(x-y))))
        return gram_matrix

def sinc(X,Y):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    sum = np.sum(np.absolute(x-y))
                    gram_matrix[i, j] = np.sinc(sum)
        return gram_matrix

def sinc2(X,Y):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    sum = np.sum(np.absolute(x-y))**2
                    gram_matrix[i, j] = np.sinc(sum)
        return gram_matrix

def quadratic(X,Y,c):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    sum = np.sum(np.absolute(x-y))**2
                    gram_matrix[i, j] = 1 - sum/(sum + c)
        return gram_matrix

def multiquadric(X,Y,c):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    sum = np.sum(np.absolute(x-y))**2
                    gram_matrix[i, j] = - np.sqrt(sum + c*c)
        return gram_matrix

def inverse_multiquadric(X,Y,c):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    sum = np.sum(np.absolute(x-y))**2
                    gram_matrix[i, j] =  1/np.sqrt(sum + c*c)
        return gram_matrix

kernels = ['laplacean', 'laplacean', 'sinc', 'sinc', 'sin2', 'quadratic', 'quadratic', 'multiquadric', 'multiquadric', 'inverse_multiquadric']

def get_kernel(i, x, y, param):
    return {
    1: lambda x,y,param: laplacean(x,y,param),
    2: lambda x,y,param: sinc(x,y),
    4: lambda x,y,param: sinc2(x,y),
    5: lambda x,y,param: quadratic(x,y,param),
    6: lambda x,y,param: quadratic(x,y,param),
    7: lambda x,y,param: multiquadric(x,y,param),
    8: lambda x,y,param: multiquadric(x,y,param),
    9: lambda x,y,param: inverse_multiquadric(x,y,param),
    3: lambda x,y,param: sinc(x,y),
    0: lambda x,y,param: laplacean(x,y,param),
    }[i](x,y,param)

for i in range(0,4):
    f.write("Zbior " + test[i] + ' ' + train[i] + '\n')
    f.write("kernel; c/gamma; tol; Wektory podpierające; jakosc zbioru trenujacego; jakosc zbiory testowego \n")
    X, y = getData(sciezka + train[i])
    X_test, y_test = getData(sciezka + test[i])
    for j in range(0,10):
        gram=get_kernel(j,X,X,10)
        clf = SVC(C=10, kernel='precomputed', tol=0.000001).fit(gram, y)
        print('Liczba wektorow podpierajacych: ', np.sum(clf.n_support_))
        print('Jakosc klasyfikacji zbioru trenujacego: ', clf.score(gram, y))
        gram_test=rbf_kernel(X_test,X,10)
        print('Test gram calculated')
        print('Jakosc klasyfikacji zbioru testowego: ', clf.score(gram_test,y_test))
        cpredicted = clf.predict(gram_test)
        f.write(kernels[j])
        f.write(';')
        f.write(str(params[j]))
        f.write(';')
        f.write(str(0.000001))
        f.write(';')
        f.write(str(np.sum(clf.n_support_)))
        f.write(';')
        f.write(str(clf.score(gram,y)))
        f.write(';')
        f.write(str(clf.score(gram_test,y_test)))
        f.write('\n')
f.close()
#for i in range (0,len(cpredicted)-1):
#    print(cpredicted[i],y_test[i])

