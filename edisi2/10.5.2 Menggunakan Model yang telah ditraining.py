#mengimport library yang digunakan
from keras.models import load_model
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

#Membaca gambar menggunakan library PIL, pembaca juga
#dapat menggunakan library lain, seperti OpenCV
gambar1= Image.open("dataset/mnist/testing/0/img_108.jpg")
gambar2= Image.open("dataset/mnist/testing/1/img_0.jpg")
#mengubah ke ndarray numpy
gambar1 = np.asarray(gambar1)
gambar2 = np.asarray(gambar2)

#memanggil model yang telah ditraining sebelumnya
model = load_model('./modelLeNet5.h5')

#memprediksi gambar dg modelnya
#ingat, intensitas perlu dibagi 255 karena saat training jg dibagi 255
#kemudian format input harus (batch size,tinggi gambar, lebar, depth)
#oleh karena itu, kita gunakan reshape seperti di bawah
pred1 = model.predict_classes((gambar1/255).reshape((1,28,28,1)))
pred2 = model.predict_classes((gambar2/255).reshape((1,28,28,1)))

#mengeplot gambar beserta hasil prediksi
plt.figure('gambar1')
plt.imshow(gambar1,cmap='gray')
plt.title('pred:'+str(pred1[0]), fontsize=22)
print("prediksi gambar1:", pred1[0])

plt.figure('gambar2')
plt.imshow(gambar2,cmap='gray')
plt.title('pred:'+str(pred2[0]), fontsize=22)
print("prediksi gambar2:", pred2[0])

plt.show()
