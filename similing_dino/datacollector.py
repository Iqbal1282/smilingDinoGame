import cv2

vid = cv2.VideoCapture(0)
count =0 

while True:
	_, frame = vid.read()

	frame = cv2.resize(frame, (600,600)) # frame = [600,600, 3]
	crop = frame[50: 400,200:500]

	cv2.imwrite('./data/open/k_%d.jpg'%(count), crop)
	count+=1


	cv2.rectangle(frame, (200, 50),(500, 400), (255,0,0), 2)


	cv2.imshow('frame', frame)
	cv2.imshow('crop', crop)


	if cv2.waitKey(1) == ord('q'):
		cv2.destroyAllWindows()
		break


	print(count)
	if count == 5:
		cv2.destroyAllWindows()
		break


