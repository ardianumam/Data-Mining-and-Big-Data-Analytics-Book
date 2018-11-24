#mengimport library yang dibutuhkan
#mengimport numpy, pandas dan keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

#membaca dataset dari file ke pandas dataFrame
irisDataset = pd.read_csv('dataset/klasifikasi_dataset_iris.csv',
                          delimiter=',', header=0)
#mengubah kelas (kolom "Species") dari string ke unique-integer
irisDataset["Species"] = pd.factorize(irisDataset.Species)[0]
#menghapus kolom "Id"
irisDataset = irisDataset.drop(labels="Id", axis=1)

#mengubah dataframe ke array numpy
irisDataset = irisDataset.as_matrix()

#membagi dataset, 40 baris data untuk training dan
#20 baris data untuk testing
dataTraining = np.concatenate((irisDataset[0:40,:],
                              irisDataset[50:90,:]), axis=0)
dataTesting = np.concatenate((irisDataset[40:50,:],
                             irisDataset[90:100,:]), axis=0)
#memecah dataset ke input dan label
inputTraining = dataTraining[:,0:4]
inputTesting = dataTesting[:,0:4]
labelTraining = dataTraining[:,4]
labelTesting = dataTesting[:, 4]
labelTraining = to_categorical(labelTraining,
                              num_classes=2)

#mendefinisikan klasifier NN
inputSize = inputTraining.shape[1]
model = Sequential()
model.add(Dense(units=10, input_dim=inputSize))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('relu'))
model.add(Dense(units=2))
model.add(Activation('sigmoid'))
model.compile(loss='categorical_crossentropy',
             optimizer='sgd',
             metrics=['accuracy'])

#train NN
model.fit(inputTraining, labelTraining,
         epochs=400, batch_size=5)

#memprediksi data testing dengan model NN yg sdh ditraining
hasilPrediksi = model.predict_classes(inputTesting,
                                     batch_size=1)

#menghitung akurasi
print("\n-------------------------------------------")
print("label sebenarnya ", labelTesting)
print("hasil prediksi ", hasilPrediksi)
prediksiBenar = (hasilPrediksi == labelTesting).sum()
prediksiSalah = (hasilPrediksi != labelTesting).sum()
print("prediksi benar: ", prediksiBenar, " data")
print("prediksi salah: ", prediksiSalah, " data")
print("akurasi: ", prediksiBenar/(prediksiBenar+prediksiSalah)
     * 100, "%")
