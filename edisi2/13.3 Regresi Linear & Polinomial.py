#import numpy dan matplotlib
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt

# data yang diberikan
x = np.array([0, 1.2, 2, 2.8, 4.3, 7, 7.8, 9])
y_aktual = np.array(([3, 8, 10, 9.5, 1.5, -8, -3.4, 2.3]))

#membuat fungsi untuk menghitung koefisien regresi
def hitungKoefisienRegresi(x, n): # x = variabel input, n = orde model
    #membuat desain matriks
    X = np.ones((x.shape[0], n+1))
    for i in range(0, x.shape[0]):
        for j in range(0, n+1):
            X[i, j] = pow(x[i],(j))
    #hitung koefisien regresi b = (X^TX)^-1 X^Ty (Persamaan 13.15)
    b = inv(np.transpose(X).dot(X)).dot(np.transpose(X).dot(y_aktual))
    return X, b

#hitung prediksi output y menggunakan koefisien b
#dengan regresi linear, yaitu orde n = 1
X, b = hitungKoefisienRegresi(x, 1)
y_prediksi_liear = X.dot(b)

#dengan regresi polinomial, misal orde n = 4
X, b = hitungKoefisienRegresi(x, 4)
y_prediksi_polim = X.dot(b)

#plot data
plt.scatter(x, y_aktual, marker="o", label = "y aktual")
plt.plot(x, y_prediksi_liear, "--", label = "regresi linear")
plt.plot(x, y_prediksi_polim, label = "regresi polinomial")
plt.legend()
plt.show()