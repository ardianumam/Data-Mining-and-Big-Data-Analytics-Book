#import numpy, matplotlib dan scikit-learn
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

#data harga tepung yg kita gunakan
data = np.array([[1,107.1],[2,113.5],[3,112.7],[4,114.7],[5,123.4],
                 [6,123.6],[7,116.3],[8,118.5],[9,119.8],[10,120.3],
                 [11,127.4],[12,125.1],[13,127.6],[14,129],[15,24.6],
                 [16,134.1],[17,146.5],[18,171.2],[19,178.6],
                 [20,172.2],[21,171.5],[22,163.6]])

#membagi data ke input data dan output data (label)
input_training=data[0:19,0].reshape((-1,1))
output_training=data[0:19,1].ravel()
input_testing=data[19:21,0].reshape((-1,1))
output_testing=data[19:21,1].ravel()

#mendefinisikan SVR dengan kernel rbf dimana gamma = 0.06
#dengan konstanta C sebesar 1e3
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.06)
#mentraining model SVR
model = svr_rbf.fit(input_training, output_training)
#memprediksi denga model yg sudah ditraining
y_prediksi_testing = model.predict(input_testing)
y_prediksi_training = model.predict(input_training)

#mengeplot data
plt.scatter(data[0:21,0].ravel(), data[0:21,1].ravel(),
            marker="o",label="data asli")
plt.plot(input_training.ravel(), y_prediksi_training,
         label="fit model training")
plt.plot(input_testing.ravel(), y_prediksi_testing, "--",
         label="prediksi data testing")
plt.ylim((data[:,1].min()-10,data[:,1].max()+10))
plt.legend()
plt.show()