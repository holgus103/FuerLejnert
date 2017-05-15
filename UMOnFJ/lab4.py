import numpy as np
from preprocessing import getData
import kernelFunctions as kf
from classifiers import MOC4, MOC16, OneVsAll4, OneVsAll16

test = ['test3.csv', 'test33.csv', 'test4.csv', 'test5.csv']
train = ['train3.csv', 'train33.csv', 'train4.csv', 'train5.csv']
cores = ['rbf', 'rbf', 'linear', 'linear', 'poly', 'poly', 'sigmoid', 'sigmoid', 'precomputed', 'precomputed', 'precomputed', 'precomputed']
sciezka = './dane/'
c = [5, 10]
tols = [0.001, 0.0001, 0.0001, 0.0001]

def writeHeader(f):
    for val in ['kernel;', 'c;', 'tolerance;', 'jakosc zbiory testowego (MOC);', "jakosc zbioru testowego(AllvsOne);\n"]:
        f.write(val)

def writeRow(kernel, c, tolerance, mocVal, avoValue):
    for val in [kernel, c, tolerance, mocVal, avoValue ]:
        f.write(str(val))
        f.write(";")
    f.write("\n")

def iteration(j, train_mapped, test_mapped):
    if(j< 3):
        resMoc = MOC4(c[j%2], cores[j], tols[j%4], train_mapped, test_mapped, y_train, y_test)
        resOvo = OneVsAll4(c[j%2], cores[j], tols[j%4], train_mapped, test_mapped, y_train, y_test)
    else:
        resMoc = MOC16(c[j%2], cores[j], tols[j%4], train_mapped, test_mapped, y_train, y_test)
        resOvo = OneVsAll16(c[j%2], cores[j], tols[j%4], train_mapped, test_mapped, y_train, y_test)
    writeRow(cores[j], c[j%2], tols[j%4], resMoc, resOvo)

f = open('results.csv', 'w')
#for each data set
kernels = [kf.laplacean, kf.sinc, kf.quadratic, kf.multiquadric]
for i in range(0, 4):
    x_train, y_train = getData(sciezka + train[i])
    x_test, y_test = getData(sciezka + test[i])
    #append header to file
    f.write(test[i])
    f.write('\n')
    writeHeader(f)
    #for each configuration
    for j in range(0, 8):
        iteration(j, x_train, x_test)
    for j in range(8, 12):
        iteration(j,kernels[j-8](x_train, x_train, 10), kernels[j-8](x_test, x_test, 10))





f.close()
