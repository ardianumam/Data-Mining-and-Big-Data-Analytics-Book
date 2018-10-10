#mengimport numpy, pytlab, intertools dan scikit-learn
import numpy as np
import pylab
import itertools
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

p=2 #ukuran dimensi tujuan

#membaca dataset MNIST dari file
X = np.loadtxt('dataset/mnist2500_X.txt')
labels = np.loadtxt('dataset/mnist2500_labels.txt')
#mendefinisikan LDA
clf = LinearDiscriminantAnalysis(n_components=2)
clf.fit(X, labels)
#mentransform data asli ke data tertransformasi menggunakan LDA
Y=clf.transform(X)
#mengeplot data tertransformasi
marker = itertools.cycle(('.','1','v', '^', '<','>','s','2','+','x'))
for label in np.unique(labels):
    pylab.scatter(np.ravel(Y[labels == label, 0]), np.ravel(Y[labels == label, 1]),
                  label=label, s=30, marker=marker.__next__())
pylab.legend()
pylab.show()
