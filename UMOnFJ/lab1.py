import numpy as np
from preprocessing import getData
from sklearn.svm import SVC

f = open('results.csv', 'w')


sciezka = './'
train = ['train1.csv', 'train11.csv', 'train2.csv', 'train22.csv' ]
test = ['test1.csv', 'test11.csv', 'test2.csv', 'test22.csv' ]
kernels = ['rbf', 'rbf', 'linear', 'linear', 'linear', 'poly', 'poly', 'poly', 'sigmoid', 'sigmoid'] 
cs = [10, 15, 10, 15, 20, 10, 15, 20, 1, 5]
tols = [0.000001, 0.000001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]
gammas = [10, 15, 10, 15, 20, 5, 10, 15, 5, 10]

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
                    gram_matrix[i, j] = np.sinc(sum)/sum
        return gram_matrix

def sinc2(X,Y):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    sum = np.sum(np.absolute(x-y))**2
                    gram_matrix[i, j] = np.sinc(sum)/sum
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
    3: lambda x,y,param: sinc(x,y,param),
    0: lambda x,y,param: laplacean(x,y,param),
    }[i](x,y,param)


print('Uzywam sciezki: ' + sciezka)
for i in range(0,4):
    f.write("Zbior " + test[i] + ' ' + train[i] + '\n')
    f.write("kernel; c; tol; gamma; wektory; jakosc zbioru trenujacego; jakosc zbiory testowego \n")
    for j in range(0,10):
        xlearn, clearn = getData(sciezka + train[i])
        xtest, ctest = getData(sciezka + test[i])

        clf = SVC(C=cs[j], kernel=kernels[j], gamma=gammas[i], tol=tols[j]).fit(xlearn,clearn)

        print('Liczba wektorow podpierajacych: ', np.sum(clf.n_support_))
        print('Jakosc klasyfikacji zbioru trenujacego: ', clf.score(xlearn,clearn))
        print('Jakosc klasyfikacji zbioru testowego: ', clf.score(xtest,ctest))

        cpredicted = clf.predict(xtest)
        print('Porownanie predykcji z faktycznymi kategoriami:')
        f.write(kernels[j])
        f.write(';')
        f.write(str(cs[j]))
        f.write(';')
        f.write(str(tols[j]))
        f.write(';')
        f.write(str(gammas[j]))
        f.write(';')
        f.write(str(np.sum(clf.n_support_)))
        f.write(';')
        f.write(str(clf.score(xlearn,clearn)))
        f.write(';')
        f.write(str(clf.score(xtest,ctest)))
        f.write('\n')
f.close()


