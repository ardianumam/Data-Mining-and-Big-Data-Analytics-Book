# Kode ini hanya untuk mengeplot data pada Contoh 1
# Untuk kode regresi, dapat dilihat di kode selanjutnya

#import numpy dan matplotlib
import numpy as np
from matplotlib import pyplot as plt

#data harga rumah Tabel 13.1
x = np.array([1400,1600,1700,1875,1100,1550,2350,2450,1425,1700])
y_aktual = np.array([245,312,279,308,199,219,405,324,319,255])
#prediksi regresi linear
y_prediksi = 98.2483 + 0.1098 * x
#plot ke grafik
plt.scatter(x, y_aktual, marker="o", label = "y aktual")
plt.plot(x, y_prediksi, label = "model regresi linear")
plt.legend()
plt.show()

