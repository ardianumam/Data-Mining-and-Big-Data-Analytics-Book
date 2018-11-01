#import numpy dan matplotlib
import numpy as np
import matplotlib.pyplot as plt

#membuat fungsi PCA
def pca(input, dimensi):
    (n, d) = input.shape
    #centering (mengurangi dg nilai mean by row datanya)
    mean_data = np.mean(input, 0)#mean data by row
    mean_data_n = np.tile(mean_data, (n, 1))#diulang sebanyak rownya
    input = input - mean_data_n#dikurangkan dengan data ori
    #PCA dilakukan dengan eigen decomposition dari matriks kovarians
    matriks_kovarians = np.dot(input.T, input)/n
    (lamda, v) = np.linalg.eig(matriks_kovarians)
    #matriks proyeksi adalah n-eigenvector (v)
    #dg n-eigenvalue (lamda) terbesar,
    matriks_proyeksi = v[:, 0:dimensi]
    #data input diproyeksikan dengan cara dikalikan (dot product)
    #dengan matriks proyeksi
    Y = np.dot(input, matriks_proyeksi)
    #print matriks kovarians, eigenvalue dan eigenvector
    #untuk keperluan pembalajaran saja
    print("mean: ", mean_data)
    print("matriks kovarians: ", matriks_kovarians)
    print("eigenvalue: ", lamda)
    print("eigenvector: ", v)
    #mengembalikan nilai data hasil proyeksi (Y)
    return Y

#data fitur kepala orang dewasa
data_ori = np.array([[191,155],[195,149],[181,148],[183,153],[176,144],
                     [208,157],[189,150],[197,159],[188,152],[192,150],
                     [179,158],[183,147],[174,150],[190,159],[188,151],
                     [163,137],[195,155],[186,153],[181,145],[175,140],
                     [192,154],[174,143],[176,139],[197,167],[190,163]])
#memproyeksikan data ori tetap ke dua dimensi
data_terproyeksi_2dimensi = pca(data_ori,2)
#mengurangi/memproyeksikan data ori ke 1 dimensi
data_terproyeksi_1dimensi = pca(data_ori,1)

#mengeplot data
plt.scatter(data_ori[:,0], data_ori[:,1], marker="o", label="data original")
axis_1D = np.tile(2,(data_ori.shape[0])) #aksis dg 1 nilai konstant untuk keperluan visualisasi saja
plt.scatter(data_terproyeksi_1dimensi, axis_1D, marker="*", label="data PCA 1 dimensi")
plt.scatter(data_terproyeksi_2dimensi[:,0], data_terproyeksi_2dimensi[:,1], marker="^", label="data PCA 2 dimensi")
plt.xlabel("panjang kepala")
plt.ylabel("lebar kepala")
plt.legend()
plt.show()
