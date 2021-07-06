import numpy as np
import cv2
from sys import exit

class VideoStabilizer:
	def __init__(self, source, *, size=(640,480), processVar=0.1, measVar=2):
		self.stream = source
		self.frameSize = size
		self.count = 0

		self.a = 0
		self.x = 0
		self.y = 0

		self.processVar = processVar
		self.measVar = measVar

		self.Q = np.array([[self.processVar]*3])
		self.R = np.array([[self.measVar]*3])

		grab, frame = self.stream.read()
		if not grab:
			print("[VideoStabilizer] No frame is captured. Exit")
			sys.exit(1)

		self.prevFrame = cv2.resize(frame, self.frameSize)
		self.prevGray = cv2.cvtColor(self.prevFrame, cv2.COLOR_BGR2GRAY)
		self.lastRigidTransform = None

		self.K_collect = []
		self.P_collect = []
	

	def read(self):
		grab, frame = self.stream.read()
		if not grab:
			print("[VideoStabilizer] No frame is captured.")
			return False, None, None   #remove last ,None

		currentFrame = cv2.resize(frame, self.frameSize)
		currentGray = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)

		self.prevPoints = cv2.goodFeaturesToTrack(self.prevGray, 
											maxCorners=200,
											qualityLevel=0.01,
											minDistance=30,
											blockSize=3)         #change max corners=10
		#feature extraction using good features to track
		#and bracket are arguments are max number of features
		#min distance is the min distance b/w two central pixels of the two features detected
		#block size of features

		currentPoints, status, err = cv2.calcOpticalFlowPyrLK(self.prevGray,
														currentGray,
														self.prevPoints,
														None)

                #pyramidalLK model to detect optical flow to compare features from previous frame to current frame's features
		
		assert self.prevPoints.shape == currentPoints.shape #look this up

		idx = np.where(status == 1)[0]         #status = 1 means succesfully current point detected
		self.prevPoints = self.prevPoints[idx]
		currentPoints = currentPoints[idx]   #previous frame features and current frame features are compared to see if they exist in both frames and those
		#which arent there are excluded

		m, inliers = cv2.estimateAffinePartial2D(self.prevPoints, currentPoints) #m is rigid transform.

###################################################################################################################################################################

		#diff b/w rigid and affine. rigid transform only looks at x and y axes. affine looks at all three axes (z means resizing)
		#affine partial2D is rigid transform

		if m is None:
			m = self.lastRigidTransform

		#m is homography matrix(look this up). how is your current frame positioned wrt previous frame

		dx = m[0, 2]
		dy = m[1, 2]

		#dx and dy show diff b/w x and y planes of the two frames

		da = np.arctan2(m[1, 0], m[0, 0]) #tells angular movement between two frames

		self.x += dx
		self.y += dy
		self.a += da

		#updating the previously defined variables. x, y and a are the terms of movement in these parameters

		Z = np.array([[self.x, self.y, self.a]], dtype="float")

               #now applying kalman filter

		if self.count == 0:
			# initialization
			self.X_estimate = np.zeros((1,3), dtype="float")
			self.P_estimate = np.ones((1,3), dtype="float")
		else:
			# extrapolation
			X_predict = self.X_estimate
			P_predict = self.P_estimate + self.Q

			# update state
			K = P_predict / (P_predict + self.R)
			self.X_estimate = X_predict + K * (Z - X_predict)
			self.P_estimate = (np.ones((1,3), dtype="float") - K) * P_predict

			self.K_collect.append(K)
			self.P_collect.append(self.P_estimate)

		diff_x = self.X_estimate[0,0] - self.x
		diff_y = self.X_estimate[0,1] - self.y
		diff_a = self.X_estimate[0,2] - self.a

		dx += diff_x
		dy += diff_y
		da += diff_a

               #making a new homography matrix

		m = np.zeros((2,3), dtype="float")
		m[0,0] = np.cos(da)
		m[0,1] = -np.sin(da)
		m[1,0] = np.sin(da)
		m[1,1] = np.cos(da)
		m[0,2] = dx
		m[1,2] = dy

                #using the new m matrix and we multiply it to the currennt frame matrix
		#self.framesize to limit the frame if it turned from the centre
		#self.fixborder to fix all the black spaces
		#warping is moving the frame based on the m matrix
                 
		frame_stabilized = cv2.warpAffine(self.prevFrame, m, self.frameSize)

		frame_stabilized = self.fixBorder(frame_stabilized)

        
		
####################################################################################################################################################
                #count is the number of frames been stablilized

#######################################################################################################################################
		self.prevGray = currentGray
		self.prevFrame = currentFrame
##############################################################################################################################################
		self.lastRigidTransform = m

		self.count += 1


		return True, currentFrame, frame_stabilized #remove frame stabilized





###############################################################################################################################################################

         #fixborder to fix all the black areas. 100% -> goes to 110% to cover up all the black areas.
       
	def fixBorder(self, frame):
		s = frame.shape
		# Scale the image 10% without moving the center
		T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.1)
		frame = cv2.warpAffine(frame, T, (s[1], s[0]))
		return frame

	def showGraph(self):
		import matplotlib.pyplot as plt 

		self.K_collect = np.array(self.K_collect)
		self.P_collect = np.array(self.P_collect)

		plt.subplot(2,1,1)
		plt.plot(range(self.K_collect.shape[0]), self.K_collect[...,0], color='r')
		plt.title("Kalman gain")
		plt.subplot(2,1,2)
		plt.plot(range(self.P_collect.shape[0]), self.P_collect[...,2], color='b')
		plt.title("Estimate uncertainty")
		plt.show()



#################################################################################################################################################################
