#import numpy dan pandas
import numpy as np
import pandas as pd

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
#mengubah array ke bentuk matriks
inputTraining = np.matrix(inputTraining)
inputTesting = np.matrix(inputTesting)

#fungsi untuk membuat desain matriks X
def buatMatriksDesainX(inputDataset):
    bias = np.full((inputDataset.shape[0],1),1.0); bias = np.matrix(bias)
    X = np.concatenate((bias, inputDataset), axis=1)
    X = np.matrix(X)
    return X

#fungsi sigmoid pada regresi logistik, lihat Persamaan (12.24)
def outputSigmoid(W, inputDataset):
    hasilSigmoid = np.ndarray(shape = (0, 1))
    for i in range(0,inputDataset.shape[0]):
        singleBias = np.full((1,1),1.0)
        x = np.concatenate((singleBias,inputDataset[i,:].T), axis=0)
        fungsiSigmoid = 1.0/(1.0 + np.exp(-np.transpose(W).dot(x)))
        hasilSigmoid = np.concatenate((hasilSigmoid,
                                       fungsiSigmoid), axis=0)
    hasilSigmoid = np.matrix(hasilSigmoid)
    return hasilSigmoid

#menginisiasi nilai W dan F(W'x)
initial_W = np.full((inputTesting.shape[1] + 1, 1), 0.1)
outSigm = outputSigmoid(initial_W, inputTraining)

#fungsi gradient descent
def gradDescent(initialWeight, inputDataset, label,
                step, iteration, tolerance):
    w_lama = initialWeight
    designMat = buatMatriksDesainX(inputDataset)
    label = np.matrix(label).T
    for i in range(0, iteration):
        outSigm = outputSigmoid(w_lama, inputDataset)
        #update W menggunakan Persamaan (12.23)
        w_baru = w_lama - step * np.transpose(designMat).dot(
            np.subtract(outSigm,label))
        deltaW = np.subtract(w_baru, w_lama)
        delta = np.abs(deltaW[:,:].sum())
        if delta < tolerance:
            break
        w_lama = w_baru
    return w_baru

#definisikan hyperparameter dari gradient descentnya
tolerance = 0.00001
konstantaStep = 0.3
banyaknyaIterasi = 400
w = gradDescent(initial_W, inputTraining, labelTraining, konstantaStep,
                banyaknyaIterasi, tolerance)

#testing: masukkan nilai W dan input x-nya ke fungsi sigmoid kembali
#kemudian bulatkan (agar <0,5 masuk ke kelas 1 (label=0)
#dan >0,5 masuk ke kelas 2 (label=1))
hasilPrediksi = np.round(outputSigmoid(w, inputTesting))
#mengubah matriks ke 1D array
hasilPrediksi = np.asarray(hasilPrediksi).ravel()

#menghitung akurasi
print("label sebenarnya ", labelTesting)
print("hasil prediksi ", hasilPrediksi)
prediksiBenar = (hasilPrediksi == labelTesting).sum()
prediksiSalah = (hasilPrediksi != labelTesting).sum()
print("prediksi benar: ", prediksiBenar, " data")
print("prediksi salah: ", prediksiSalah, " data")
print("akurasi: ", prediksiBenar/(prediksiBenar+prediksiSalah) * 100, "%")