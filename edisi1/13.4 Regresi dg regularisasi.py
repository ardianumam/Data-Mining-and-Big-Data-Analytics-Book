#import numpy dan matplotlib
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt

# data yang diberikan
x = np.array([0, 1.2, 2, 2.8, 4.3, 7, 7.8, 9])
y_aktual = np.array(([3, 8, 10, 9.5, 1.5, -8, -3.4, 2.3]))

#membuat fungsi untuk menghitung koefisien regresi
def hitungKoefisienRegresi(x, n, lamda): # x adalah variabel input dan n = orde model
    #membuat desain matriks
    X = np.ones((x.shape[0], n+1))
    for i in range(0, x.shape[0]):
        for j in range(0, n+1):
            X[i, j] = pow(x[i],(j))
    #hitung koefisien regresi b = (X'X+lamda*I)^-1 X'y (persamaan 13.17)
    b = inv(np.add(np.transpose(X).dot(X),
                   lamda*np.identity(n+1))).dot(np.transpose(X).dot(y_aktual))
    return X, b

#prediksi output y dg regresi polinomial + regularisasi
# misal orde n = 4 dan konstanta regularisasi lamda = 1
X, b = hitungKoefisienRegresi(x, 4, 1)
y_prediksi_polim_regu = X.dot(b)
#prediksi output y dg regresi polinomial tanpa regularisasi
# misal orde n = 4, sehingga konstanta regularisasi lamda = 0
X, b = hitungKoefisienRegresi(x, 4, 0)
y_prediksi_polim_tanpa_regu = X.dot(b)

#plot data
plt.scatter(x, y_aktual, marker="o", label = "y aktual")
plt.plot(x, y_prediksi_polim_regu, label = "regresi polinomial+regu")
plt.plot(x, y_prediksi_polim_tanpa_regu, "--", label = "regresi polinomial tanpa regu")
plt.legend()
plt.show()
