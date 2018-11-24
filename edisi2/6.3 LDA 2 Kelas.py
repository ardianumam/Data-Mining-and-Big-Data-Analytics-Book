#import numpy
import numpy as np
from numpy import genfromtxt

#membaca data dari file
Xole = genfromtxt('dataset/6.3_HalticaOleracea.csv', delimiter=',')
Xcar = genfromtxt('dataset/6.3_HalticaCarduorum.csv', delimiter=',')

#hitung matrik kovarians kelas 1 (Haltica oleracea)
#dan kelas 2 (Haltica carduorum)
XoleCentered = Xole - Xole.mean(axis=0)
XcarCentered = Xcar - Xcar.mean(axis=0)
Sole = np.dot(XoleCentered.T, XoleCentered) / (Xole.shape[0] - 1)
Scar = np.dot(XcarCentered.T, XcarCentered) / (Xcar.shape[0] - 1)
print("Sole: ", Sole)
print("Scar: ", Scar)

#hitung matrik kovarians S (gabungan)
S = ((Xole.shape[0] - 1) * Sole + (Xcar.shape[0] - 1) * Scar)\
    / ((Xole.shape[0] - 1) + (Xcar.shape[0] - 1))
print("S: ", S)

#hitung matrik projeksi a (koefisien LDA)
meanXol = Xole.mean(axis=0); meanXol=np.asmatrix(meanXol).T
meanXcar = Xcar.mean(axis=0); meanXcar=np.asmatrix(meanXcar).T
a = np.dot(np.linalg.inv(S), np.subtract(meanXol, meanXcar))
a = np.asmatrix(a)

#hitung treshold
thresh = 0.5 * np.dot((np.add(meanXol, meanXcar)).T, a)
print("matriks proyeksi a: ", a)
print("konstanta treshold: ", thresh)

#proyeksikan data dengan matriks proyeksi a
XoleLDA = np.dot(Xole,a)
XcarLDA = np.dot(Xcar,a)
print("Haltica Oleracea setelah LDA: ", XoleLDA)
print("Haltica Carduorum setelah LDA: ", XcarLDA)

#terapkan threshold untuk pemisahan kelas
XoleLDA_prediksi = np.full((XoleLDA.shape),0)
XoleLDA_prediksi[XoleLDA>thresh]=1#jika lbh dari thresh, masuk ke kelas 1
XoleLDA_prediksi[XoleLDA<=thresh]=2#jika krg dari atw sama, masuk ke kelas 2

XcarLDA_prediksi = np.full((XcarLDA.shape),0)
XcarLDA_prediksi[XcarLDA>thresh]=1#jika lbh dari thresh, masuk ke kelas 1
XcarLDA_prediksi[XcarLDA<=thresh]=2#jika krg dari/sama, masuk ke kelas 2
print("XoleLDA_prediksi:", "\n", XoleLDA_prediksi)
print("XcarLDA_prediksi:", "\n", XcarLDA_prediksi)