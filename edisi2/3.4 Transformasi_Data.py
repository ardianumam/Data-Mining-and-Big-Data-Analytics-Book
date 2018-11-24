#import numpy
import numpy as np

#mendefinisikan matrik x, bebas, untuk contoh saja
x = np.matrix([[-20,23,5],[4,-8,15]])

#centering (data asli dikurangi nilai reratanya)
#sesuai Persamaan 3.1
x_centering = x - x.mean()
print("x_centering: ", x_centering, "\n")

#standarisasi sesuai Persamaan 3.4
x_standarisasi = (x-x.mean()) / x.std()
print("x_standarisasi: ", x_standarisasi)
print("mean x_standarisasi: ", x_standarisasi.mean())
print("varian x_standarisasi: ", x_standarisasi.var(), "\n")

#scaling ke range 0-1, sesuai Persamaan 3.5
BA = 1; BB = 0 #BA=batas atas, BB = batas bawah
x_scaling = (x - x.min()) / (x.max()-x.min()) * (BA-BB) + BB
print("x_scaling: ", x_scaling)

