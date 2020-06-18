'''
1. Making model
2. Taking data  
3. Connecting to the dino game 

'''

import tensorflow as tf 
from utils import load_data
from sklearn.model_selection import train_test_split





feature, labels = load_data()

x_train, x_test , y_train, y_test = train_test_split(feature, labels,
	test_size = 0.1 )






input = tf.keras.layers.Input((100,100,3))

x = tf.keras.layers.Conv2D(64,kernel_size=(1,1),activation='relu',
	padding='same')(input)
x = tf.keras.layers.MaxPool2D(pool_size = (2,2), strides=(2,2),
	padding='same')(x)

x = tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu',
	padding='same', use_bias= False)(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2),
	padding='same')(x)


x = tf.keras.layers.Conv2D(256,kernel_size=(5,5), activation='relu',
	padding='same')(x)

x = tf.keras.layers.MaxPool2D(pool_size=(5,5), strides=(2,2),
	padding='same')(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(512, kernel_size=(3,3),activation='relu',
	padding='same')(x)

y = x


x = tf.keras.layers.Conv2D(256, kernel_size=(3,3), activation='relu',
	padding='same')(x)

x = tf.keras.layers.Conv2D(512, kernel_size=(3,3), activation='relu',
	padding='same')(x)


x = x+y 

x = tf.keras.layers.Conv2D(1024, kernel_size=(3,3), activation='relu',
	padding='same')(x)

x = tf.keras.layers.Flatten()(x)

#x = tf.keras.layers.Dense(1024, activation='relu')(x)
#x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)

output = tf.keras.layers.Dense(1, activation='sigmoid')(x)


model = tf.keras.Model(input, output)

model.summary()

model.compile(loss= 'binary_crossentropy',
	optimizer = 'adadelta',
	metrics=['accuracy'])


model.fit(x_train, y_train, epochs = 1,
	batch_size = 100,
	validation_data = (x_test, y_test))


model.save('neuralModel.h5')



