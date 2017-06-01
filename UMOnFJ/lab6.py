#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from preprocessing import getData
from sklearn.svm import SVR
import kernelFunctions as kf

#import matplotlib.pyplot as plt
#from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D

sciezka = './dane/'
print('Uzywam sciezki: ' + sciezka)
sets = [['train6.csv', 'test6.csv'], ['train7.csv', 'test7.csv']]

# each entry is a list of C, gamma and degree
configs = [[1, 0.1, 2], [5, 0.2, 4], [20, 0.4, 8]]
for set_num in range(0, len(sets)):
    xlearn, rlearn = getData(sciezka + sets[set_num][0])
    xtest, rtest = getData(sciezka + sets[set_num][1])
    print(sets[set_num])
    for c_num in range(0, len(configs)):
        svr_rbf = SVR(kernel='rbf', C=configs[c_num][0], gamma=configs[c_num][1])
        svr_lin = SVR(kernel='linear', C=configs[c_num][0])
        svr_poly = SVR(kernel='poly', C=configs[c_num][0], degree=configs[c_num][2])    
        r_laplacean = SVR(kernel ='precomputed').fit(kf.laplacean(xlearn, xlearn, configs[c_num][1]), rlearn).predict(kf.laplacean(xtest, xlearn, configs[c_num][1]))
        r_sinc = SVR(kernel ='precomputed').fit(kf.sinc(xlearn, xlearn, configs[c_num][0]), rlearn).predict(kf.sinc(xtest, xlearn, configs[c_num][0]))
        r_quadratic = SVR(kernel ='precomputed').fit(kf.quadratic(xlearn, xlearn, configs[c_num][0]), rlearn).predict(kf.quadratic(xtest, xlearn, configs[c_num][0]))
        r_multiquadric = SVR(kernel ='precomputed').fit(kf.multiquadric(xlearn, xlearn, configs[c_num][0]), rlearn).predict(kf.multiquadric(xtest, xlearn, configs[c_num][0]))
        r_rbf = svr_rbf.fit(xlearn, rlearn).predict(xtest)
        r_lin = svr_lin.fit(xlearn, rlearn).predict(xtest)
        r_poly = svr_poly.fit(xlearn, rlearn).predict(xtest)

        print('Tu liczymy błąd aproksymacji:')
        blad_rbf=0;
        for i in range (0,len(r_rbf)-1):
            blad_rbf=blad_rbf+(r_rbf[i]-rtest[i])**2
        print('Błąd regresji dla funkcji rbf: ', blad_rbf)

        blad_lin=0;
        for i in range (0,len(r_lin)-1):
            blad_lin=blad_lin+(r_lin[i]-rtest[i])**2
        print('Błąd regresji dla funkcji liniowej: ', blad_lin)

        blad_poly=0;
        for i in range (0,len(r_poly)-1):
            blad_poly=blad_poly+(r_poly[i]-rtest[i])**2
        print('Błąd regresji dla funkcji wielomianowej: ', blad_poly)

        print('Tu liczymy błąd aproksymacji:')
        blad_rbf=0;
        for i in range (0,len(r_laplacean)-1):
            blad_rbf=blad_rbf+(r_laplacean[i]-rtest[i])**2
        print('Błąd regresji dla funkcji laplacean: ', blad_rbf)

        blad_lin=0;
        for i in range (0,len(r_sinc)-1):
            blad_lin=blad_lin+(r_sinc[i]-rtest[i])**2
        print('Błąd regresji dla funkcji sinc: ', blad_lin)

        blad_poly=0;
        for i in range (0,len(r_quadratic)-1):
            blad_poly=blad_poly+(r_quadratic[i]-rtest[i])**2
        print('Błąd regresji dla funkcji quadratic: ', blad_poly)

        blad_poly=0;
        for i in range (0,len(r_multiquadric)-1):
            blad_poly=blad_poly+(r_multiquadric[i]-rtest[i])**2
        print('Błąd regresji dla funkcji multiquadric: ', blad_poly)



#fig=plt.figure()
#axe=Axes3D(fig)
#ax = fig.add_subplot(111, projection='3d')
###Dane trenujące
#surf1 = axe.plot_surface(xlearn[0:xlearn.size,0], xlearn[0:xlearn.size,1], rlearn[0:rlearn.size], rstride=1, cstride=1, cmap=cm.coolwarm,
# linewidth=0, antialiased=False)
#fig.colorbar(surf1, shrink=0.5, aspect=5)
#plt.show()

###Ilustracja wyników testowania dla SVR z funkcją RBF:
#fig=plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#surf2 = ax.plot_surface(xtest[0:xtest.size,0], xtest[0:xtest.size,1], rtest[0:rtest.size], rstride=1, cstride=1, cmap=cm.coolwarm)
#fig.colorbar(surf2, shrink=0.5, aspect=5)
#plt.show()
##surf3 = ax.plot_surface(xtest[0:xtest.size,0], xtest[0:xtest.size,1], r_rbf[0:r_rbf.size], rstride=1, cstride=1, cmap=cm.coolwarm)
##fig.colorbar(surf3, shrink=0.5, aspect=5)
##plt.title('Support Vector Regression')
##plt.show()
