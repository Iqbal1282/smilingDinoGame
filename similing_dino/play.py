import cv2 
import tensorflow as tf 
import numpy as np 
import time 

from directKeys import PressKey , ReleaseKey


model = tf.keras.models.load_model('neuralModel.h5')

vid = cv2.VideoCapture(0)


SpaceBar = 0x20 

previous_key =0 
while True:
	_, frame = vid.read()

	frame = cv2.resize(frame, (600,600)) # frame = [600,600, 3]
	crop = frame[50: 400,200:500]
	crop = cv2.resize(crop, (100,100))
	



	crop_dim = tf.expand_dims(crop, 0)

	prediction = model.predict(crop_dim)
	print(prediction[0][0])

	if int(np.around(prediction[0][0]))==1:
		previous_key =1
		PressKey(SpaceBar)
		ReleaseKey(SpaceBar)
		time.sleep(0.005)
		print('pressed')




	cv2.rectangle(frame, (200, 50),(500, 400), (255,0,0), 2)

	cv2.imshow('frame', frame)


	if cv2.waitKey(1) == ord('q'):
		cv2.destroyAllWindows()
		break