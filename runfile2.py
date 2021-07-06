import os
from sys import exit
sys.path.append("C:/Users/KSHITIJ/Desktop/pyyyyyy")

import cv2
from degenX import VideoStabilizer



video = cv2.VideoCapture("video 2.mp4")
stabilizer = VideoStabilizer(video)

while True:
	success, _, frame = stabilizer.read()
	if not success:
		print("No frame is captured.")
		break
		
	cv2.imshow("frame", frame)
	if cv2.waitKey(20) == 27:
		break
stabilizer.showGraph()
video.release() 
cv2.destroyAllWindows() 


