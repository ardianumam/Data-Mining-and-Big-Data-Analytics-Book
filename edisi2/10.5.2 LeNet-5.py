#mengimport library yang dibutuhkan
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

#spesifikasi yang kita gunakan
img_width, img_height = 28,28 #lebar dan tinggi gambar
input_depth = 1 #1 karena kita gunakan gambar gray
train_data_dir = 'dataset/mnist/training' #folder data training
testing_data_dir = 'dataset/mnist/testing' #folder data testing
epochs = 2 #jumlah epoch training yg kita inginkan
batch_size = 5#batch size yang kita inginkan

#mendefinisikan image generator untuk training
#& testing, di mana nilai intensitas diskalakan
#ke 0-1 dg cara dibagi dengan 255
train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

#membaca data dengan ImageDataGenerator batch demi batch
#untuk data training dan data testing
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',#mode warna gray
    target_size=(img_width,img_height),#target ukuran gambar
    batch_size=batch_size,#ukuran batch
    class_mode='categorical')#categorical di Keras berarti one-hot encoding
testing_generator = test_datagen.flow_from_directory(
    testing_data_dir,
    color_mode='grayscale',
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='categorical')

#mendefinisikan format dimensi dari input image
if K.image_data_format() == 'channels_first':
    input_shape_val = (input_depth, img_width, img_height)
else:
    input_shape_val = (img_width, img_height, input_depth)

#mendefinisikan model sequential di Keras
model = Sequential()

# C1 Convolutional Layer
#menambahkan layer konvolusi dengan 6 filter, berukuran 5x5
#untuk layer pertama kita perlu definisikan bentuk inputnya
#di sini kita berikan padding='same'
model.add(Conv2D(6, (5, 5),input_shape=input_shape_val, padding='same'))
#menambahkan fungsi aktivasi relu
model.add(Activation('relu'))

# S2 Pooling Layer
#menambahkan layer max-pooling dengan filter ukuran 2x2
model.add(MaxPool2D((2, 2)))

# C3 Convolutional Layer
#menambahkan layer konvolusi dengan 16 filter, berukuran 5x5
#untuk layer selanjutnya, Keras dapat mengenali ukuran input
#di sini kita jg berikan padding='same'
model.add(Conv2D(16, (5, 5), padding='same'))
#menambahkan fungsi aktivasi relu
model.add(Activation('relu'))

# S4 Pooling Layer
#menambahkan layer max-pooling dengan filter ukuran 2x2
model.add(MaxPool2D((2, 2)))

# C5 Fully Connected Convolutional Layer
#menambahkan layer konvolusi dengan 120 filter, berukuran 5x5
#di sini kita jg berikan padding='same'
model.add(Conv2D(120, (5, 5), padding='same'))
#menambahkan fungsi aktivasi relu
model.add(Activation('relu'))

#Untuk bisa dihubungkan ke fully connected layer,
#maka hasil konvolusi di atas perlu di-flatten
#agar mendjadi array satu dimensi
model.add(Flatten())

# FC6 Fully Connected Layer
model.add(Dense(84, activation='relu'))

#Output Layer dengan softmax
model.add(Dense(train_generator.num_classes, activation='softmax'))

#Mengkompile model
model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])


#Mentraining model sekaligus memvalidasi dengan testing data
model.fit_generator(
    train_generator,#training generator kita
    #jumlah iterasi per epoch = banyaknya data / ukuran batch
    steps_per_epoch=np.floor(train_generator.n/batch_size),
    epochs=epochs,#jumlah epoch
    validation_data=testing_generator,#testing data kita
    #jumlah iterasi per epoch = banyaknya data / ukuran batch
    validation_steps=np.floor(testing_generator.n / batch_size))

print("Training sudah selesai!")
model.save('modelLeNet5.h5')
model.save_weights('weightLeNet5.h5')
print("Model sudah disimpan!")