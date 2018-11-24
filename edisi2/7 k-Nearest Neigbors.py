#mengimport numpy, pandas dan scipy
import numpy as np
import pandas as pd
from scipy import stats

#membaca dataset dari file ke pandas dataFrame
irisDataset = pd.read_csv('dataset/klasifikasi_dataset_iris.csv',
                          delimiter=',', header=0)
#mengubah kelas (kolom "Species") dari string ke unique-integer
irisDataset["Species"] = pd.factorize(irisDataset.Species)[0]
#menghapus kolom "Id"
irisDataset = irisDataset.drop(labels="Id", axis=1)

#mengubah dataframe ke array numpy
irisDataset = irisDataset.as_matrix()

#membagi dataset, 40 baris data untuk training
#dan 20 baris data untuk testing
dataTraining = np.concatenate((irisDataset[0:40,:],
                               irisDataset[50:90,:]), axis=0)
dataTesting = np.concatenate((irisDataset[40:50,:],
                              irisDataset[90:100,:]), axis=0)
#memecah dataset ke input dan label
inputTraining = dataTraining[:,0:4]
inputTesting = dataTesting[:,0:4]
labelTraining = dataTraining[:,4]
labelTesting = dataTesting[:, 4]

k=3#inputnya nilai k-Nearest Neighborsnya
#prediksi data testing menggunakan data training
matriks_prediksi_kelas = np.ndarray(shape = (0,1))
#mengkasting label ke bentuk matriks
labelTraining=np.matrix(labelTraining).T
for i in range(0,inputTesting.shape[0]):#loop semua data testing
    for j in range(0, inputTraining.shape[0]): #loop semua data training
        #hitung jarak euclidean tiap satu data testing
        #ke semua data training
        euclideanDistance = np.square(np.sum((np.tile(
            inputTesting[i,:],
            (inputTraining.shape[0],1))-inputTraining)**2,axis=1))
        #mencasting ke variabel matriks
        euclideanDistance=np.matrix(euclideanDistance).T
        # menambahkan kolom label ke matriks euclidean
        matriksEuclideanDanLabel = np.concatenate((euclideanDistance,
                                                   labelTraining), axis=1)
        #casting ke array dulu untuk disorting
        matriksEuclideanDanLabel = np.asarray(matriksEuclideanDanLabel)
        #sorting berdasarkan jarak euclidean
        matriksEuclideanDanLabelSorted = matriksEuclideanDanLabel[
            matriksEuclideanDanLabel[:, 0].argsort()]
        #mengambil k-label kelas dengan jarak euclidean plg kecil
        k_label=matriksEuclideanDanLabelSorted[0:k,1]
        #prediksi adalah kelas dengan kemunculan terbanyak dari k-label
        prediksi_kelas = np.matrix(stats.mode(k_label)[0])
    #menggabungkan semua prediksi dalam matriks
    matriks_prediksi_kelas=np.concatenate((matriks_prediksi_kelas,
                                           prediksi_kelas), axis=0)

#menghitung akurasi
matriks_prediksi_kelas=matriks_prediksi_kelas.ravel()#flatten ke 1D array
prediksiBenar = (matriks_prediksi_kelas == labelTesting).sum()
prediksiSalah = (matriks_prediksi_kelas != labelTesting).sum()
print("prediksi benar: ", prediksiBenar, " data")
print("prediksi salah: ", prediksiSalah, " data")
print("akurasi: ", prediksiBenar/(prediksiBenar+prediksiSalah) * 100, "%")