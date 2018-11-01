#import library yang dibutuhkan
import numpy as np

#data input = gerbang logika N
inputLogikaAnd = np.matrix([[1,1],[-1,1],[1,-1],[-1,-1]])
bias = np.matrix([[1],[1],[1],[1]])
#menambahkan bias 1 ke input pada kolom pertama
x = np.concatenate((bias,inputLogikaAnd), axis=1)

#inisasi bobot w dg nilai nol
w = np.full((1,3),0)

#label d
d = np.matrix([[1],[-1],[-1],[-1]])

#definisikan learning rate, misal 1
n = 1

#loop hingga konvergen
counter = 0
while(True):
    wx=w.dot(x[counter,:].T)
    sign = np.sign(wx)
    if (wx==0):
        sign = -1
    #cek apakah output sign sesuai dg label 'd'
    if (sign!=np.asscalar(d[counter])):
        #update bobot w
        w = np.add(w, n*(np.asscalar(d[counter])-sign)*x[counter,:])
    counter = counter + 1
    #reset counter ke nol jika sudah sampai ke baris data terakhir
    if (counter==(x.shape[0]-1)):
        counter = 0
    #cek apakah sudah konvergen, jika iya maka stop (break)
    finalSign = np.sign(w.dot(x.T)-1)
    if ((finalSign.T == d).all()):
        break
print("w = ", w)
print("label d = ", d)
print("prediksi = ", finalSign.T)
