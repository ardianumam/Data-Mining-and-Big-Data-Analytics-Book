# import library
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

#buat array 2D ke dalam variabel x. Data ini yang akan kita klaster.
x = np.array([[1,2],[2.5,4.5],[2,2],[4,1.5],[4,2.5]])

#lakukan hierachical klastering
Z = linkage(x)

#plot dendogram
plt.figure(figsize=(25, 10))
plt.title('Dendrogram Hierarchical Klastering')
plt.xlabel('data')
plt.ylabel('jarak')
dendrogram(Z)
plt.show()
