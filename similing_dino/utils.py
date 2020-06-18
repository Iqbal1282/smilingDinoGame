import os

import numpy as np 
import pickle 
import cv2


data_dir = './data'

categories = ['close', 'open']

data = []


def make_data():
	for category in categories:
		path = os.path.join(data_dir, category)
		label = categories.index(category)


		for image_name in os.listdir(path):
			image_path = os.path.join(path, image_name)
			image = cv2.imread(image_path)

			try:
				#image = cv2.cvtColor()
				image = cv2.resize(image, (100, 100))
				image = np.array(image)


				data.append([image, label])
			except Exception as e:
				pass

	print(len(data))

	pik = open('mouth.pickle', 'wb')
	pickle.dump(data, pik)

	pik.close()


def load_data():
	pick = open('mouth.pickle', 'rb')
	data = pickle.load(pick)
	pick.close()


	feature = []
	labels = []

	for image, label in data:
		feature.append(image)
		labels.append(label)

	feature = np.array(feature, dtype=np.float32) 
	feature = feature/255.0 

	labels = np.array(labels)

	return [feature, labels]



if __name__ == '__main__':
	make_data()







