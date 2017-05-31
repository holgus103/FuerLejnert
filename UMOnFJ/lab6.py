import numpy as np
from preprocessing import getData
from sklearn.svm import SVR

#import matplotlib.pyplot as plt
#from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D

sciezka = '../lab6/'
print('Uzywam sciezki: ' + sciezka)
xlearn, rlearn = getData(sciezka + 'train7.csv')
xtest, rtest = getData(sciezka + 'test7.csv')

svr_rbf = SVR(kernel='rbf', C=1, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1)
svr_poly = SVR(kernel='poly', C=1, degree=2)

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


