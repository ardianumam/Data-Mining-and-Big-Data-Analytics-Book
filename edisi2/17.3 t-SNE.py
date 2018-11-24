#import numpy, matplotlib, pylab dan intertools
import numpy as np
import pylab
import matplotlib.mlab as mlab
import itertools


def Hbeta(D=np.array([]), beta=1.0):
    """
        Hitung perlexity dan P-row untuk spesifik nilai presisi
        dari distribusi Gaussian
    """

    # Hitung P-row dan perlexity yang bersangkutan
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Melakukan binary search untuk mendapatkan P-value sehingga
        di tiap-tiap locality yang dimodelkan dengan fung Gaussiannya
        memiliki nilai perplexity yang relatif sama
    """

    # Inisialisasi variable
    print("Hitung pairwise distance...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Looping untuk semua data poin
    for i in range(n):

        # Print progres
        if i % 500 == 0:
            print("Menghitung P-value untuk poin %d dari %d..." % (i, n))

        # Hitung kernel Gaussian dan entropy dari presisi yang sekarang
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluasi apakah nilai perplexity dalam toleransi
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # Jika tidak, naikkan atau turunkan presisi
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Hitung ulang
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set row final dari P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Nilai rata-rata dari sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Jalankan PCA dulu untuk mengurangi dimensi asli ke ukuran dimensi 50
    """

    print("Preprocessing data dengan PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50,
         perplexity=30.0, max_iter=()):
    """
        Jalankan t-SNE dari dataset asli untuk mengurangi dimensinya
        ke ukuran no_dims
    """

    # Cek input
    if isinstance(no_dims, float):
        print("Error: array X harus bertipe float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: nomor dimensinya harus bertipe integer.")
        return -1

    # Inisialisasi variabel
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Hitung P-values (similarity dari data di dimensi asli, dimensi tinggi)
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.
    P = np.maximum(P, 1e-12)

    # Jalankan iterasi
    for iter in range(max_iter):

        # Hitung similarity data terpetakan di dimensi rendah
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Hitung gradientnya
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i],
                                      (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Lakukan update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Hitung output dari fungsi objektif (cost) saat ini
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        if iter == 100:
            P = P / 4.

    # Return solution
    return Y,P,Q


if __name__ == "__main__":
    print("Menjalankan t-SNE pada 2.500 MNIST data gambar digit angka.")
    X = np.loadtxt('dataset/mnist2500_X.txt')
    labels = np.loadtxt('dataset/mnist2500_labels.txt')
    #parameter dari t-SNE
    perplexityValue = 50.0
    no_dimensi=2
    dimensi_reduksi_dg_PCA=50
    max_iter = 1000
    #panggil fungsi t-SNE
    Y,P,Q = tsne(X, no_dimensi, dimensi_reduksi_dg_PCA,
                 perplexityValue, max_iter)
    P2500 = np.mean(P, axis=0).ravel()
    Q2500 = np.mean(Q, axis=0).ravel()
    #plot histogram dari sebaran P dan Q
    pylab.figure("histogram")
    nQ, binsQ, patchesQ = pylab.hist(Q2500, color="green", alpha=0.8,
                                     label="Q" + ",mean: " +
                                           str(np.mean(Q)) +
                                           ",var: " + str(np.var(Q)))
    nP, binsP, patchesP = pylab.hist(P2500, color="red", alpha=0.8,
                                     label="P" + ",mean: " +
                                           str(np.mean(P)) +
                                           ",var: " + str(np.var(P)))
    Qfit = mlab.normpdf(binsQ, np.mean(Q), np.var(Q))
    Pfit = mlab.normpdf(binsP, np.mean(P), np.var(P))
    pylab.plot(binsQ, Qfit, 'g--')
    pylab.plot(binsP, Pfit, 'r--')
    pylab.xlabel('nilai similarity')
    pylab.ylabel('banyaknya/frekuensi kemunculan')
    pylab.legend()
    #plot hasil data terpetakan menggunakan t-SNE
    pylab.figure("t-sne, perplexity: " + str(perplexityValue))
    marker = itertools.cycle(('.','1','v', '^', '<','>','s','2','+','x'))
    for label in np.unique(labels):
        pylab.scatter(Y[labels == label, 0], Y[labels == label, 1],
                      label=label, s=30, marker=marker.__next__())
    pylab.legend()
    pylab.show()