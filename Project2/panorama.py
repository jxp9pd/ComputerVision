import cv2, random, skimage, skimage.transform
import numpy as np
from matplotlib import pyplot as plt
#img1 corresponds to left-side of landscape, img2 corresponds to right-side
img1 = cv2.imread("/Users/johnp_000/Dropbox/UVA 2016-2017/Spring Semester/CS 4501/pics/inputA.jpg")
img2 = cv2.imread("/Users/johnp_000/Dropbox/UVA 2016-2017/Spring Semester/CS 4501/pics/inputB.jpg")
WIDTH = len(img1[0])
HEIGHT = len(img1)
#RANSAC Iterations
N = 50
print(WIDTH)
print(HEIGHT)
gray_image = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
# Apply ratio the st
good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
counter = 0
mapSet1 = []
mapSet2 = []
fullMap1 = []
fullMap2 = []

for m,n in matches:
    if (m.distance < 0.75*n.distance):
        good.append([m])
        match = m
        img1Idx = match.queryIdx
        img2Idx = match.trainIdx
        (x1,y1) = kp1[img1Idx].pt
        (x2,y2) = kp2[img2Idx].pt
        mapSet1.append((x1,y1))
        mapSet2.append((x2,y2))
    match = m
    img1Idx = match.queryIdx
    img2Idx = match.trainIdx
    (x1,y1) = kp1[img1Idx].pt
    (x2,y2) = kp2[img2Idx].pt
    fullMap1.append((x1,y1))
    fullMap2.append((x2,y2))
print ("Total number of matches: " +  str(len(matches)))
print ("My list of total matches: " + str(len(fullMap1)))
# for i in range (0,8):
# 	match = matches[i]
# 	print (type(matches[i]))
# 	img1Idx = match.queryIdx
# 	img2Idx = match.trainIdx
# 	(x1,y1) = kp1[img1Idx].pt
# 	(x2,y2) = kp2[img2Idx].pt
# 	mapSet1.append((x1,y1))
# 	mapSet2.append((x2,y2))
# 	counter+=1
#good = sorted(good, key = lambda x:x.distance)
outImage = np.zeros((HEIGHT, WIDTH))
# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,outImage)
# plt.imshow(img3),plt.show()

def homography(pixelB, H):
	xaTop = pixelB[0]*H[0][0] + pixelB[1]*H[0][1] + H[0][2]
	xaBottom=pixelB[0]*H[2][0] + pixelB[1]*H[2][1] + H[2][2]
	xa = xaTop/xaBottom
	yaTop = pixelB[0]*H[1][0] + pixelB[1]*H[1][1] + H[1][2]
	yaBottom=pixelB[0]*H[2][0] + pixelB[1]*H[2][1] + H[2][2]
	ya = yaTop/yaBottom
	return (xa,ya)

aList = [(0,0), (1,0), (0,1), (1,1)]
bList = [(1,2), (3,2), (1,4), (3,4)]

def makeHomo(aPoints, bPoints):
	a1 = aPoints[0]
	a2 = aPoints[1]
	a3 = aPoints[2]
	a4 = aPoints[3]
	
	b1 = bPoints[0]
	b2 = bPoints[1]
	b3 = bPoints[2]
	b4 = bPoints[3]

	A = [[-1*b1[0], -1*b1[1], -1, 0, 0, 0, a1[0]*b1[0], a1[0]*b1[1]],
	 [0, 0, 0, -1*b1[0], -1*b1[1], -1, a1[1]*b1[0], a1[1]*b1[1]],
	 [-1*b2[0], -1*b2[1], -1, 0, 0, 0, a2[0]*b2[0], a2[0]*b2[1]],
	 [0, 0, 0, -1*b2[0], -1*b2[1], -1, a2[1]*b2[0], a2[1]*b2[1]],
	 [-1*b3[0], -1*b3[1], -1, 0, 0, 0, a3[0]*b3[0], a3[0]*b3[1]],
	 [0, 0, 0, -1*b3[0], -1*b3[1], -1, a3[1]*b3[0], a3[1]*b3[1]],
	 	 [-1*b4[0], -1*b4[1], -1, 0, 0, 0, a4[0]*b4[0], a4[0]*b4[1]],
	 [0, 0, 0, -1*b4[0], -1*b4[1], -1, a4[1]*b4[0], a4[1]*b4[1]]]

	B = [-1*a1[0],-1*a1[1], -1*a2[0],-1*a2[1], -1*a3[0],-1*a3[1],-1*a4[0],-1*a4[1]]
	x = np.linalg.lstsq(A,B)[0]
	homography = [[x[0], x[1], x[2]],
				  [x[3], x[4], x[5]],
				  [x[6], x[7], 1]]
	return homography

# x = makeHomo(aList, bList)
# print(x)

#RANSAC
def ransac():
	randomA = []
	randomB = []
	for i in range(0,4):
		index = int(random.random()*len(mapSet1))
		randomA.append(mapSet1[index])
		randomB.append(mapSet2[index])
	H = makeHomo(randomA, randomB)
	return H
	
maxInliers = 0
bestH = np.zeros((3,3))

for n in range(0,300):
	H = ransac()
	inliers = 0
	for i in range(len(mapSet2)):
		a = homography(mapSet2[i], H)
		trueA = mapSet1[i]
		diff = ((a[0] + a[1])-(trueA[0] + trueA[1]))**2
		if (diff<2):
			inliers+=1
	if inliers>maxInliers:
		maxInliers = inliers
		bestH = H
print (bestH)
print (maxInliers)
print(len(mapSet2))

#Image mapping
def composite_warped(a, b, H):
    out_shape = (a.shape[0], 2*a.shape[1])                            # Output image (height, width)
    p = skimage.transform.ProjectiveTransform(np.linalg.inv(H))       # Inverse of homography (used for inverse warping)
    bwarp = skimage.transform.warp(b, p, output_shape=out_shape)      # Inverse warp b to a coords
    bvalid = np.zeros(b.shape, 'uint8')                               # Establish a region of interior pixels in b
    bvalid[1:-1,1:-1,:] = 255
    bmask = skimage.transform.warp(bvalid, p, output_shape=out_shape)    # Inverse warp interior pixel region to a coords
    apad = np.hstack((skimage.img_as_float(a), np.zeros(a.shape))) # Pad a with black pixels on the right
    return skimage.img_as_ubyte(np.where(bmask==1.0, bwarp, apad))    # Select either bwarp or apad based on mask
imgFinal = composite_warped(img1, img2, bestH)
plt.imshow(imgFinal),plt.show()
