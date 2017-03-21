import numpy, skimage, skimage.io, pylab, scipy.ndimage.filters, cv2

#import Images
left=skimage.io.imread('/Users/johnp_000/Dropbox/UVA 2016-2017/Spring Semester/CS 4501/pics/im0_small.png')
left=skimage.img_as_float(left)
right=skimage.io.imread('/Users/johnp_000/Dropbox/UVA 2016-2017/Spring Semester/CS 4501/pics/im1_small.png')
right=skimage.img_as_float(right)
ground = numpy.load('/Users/johnp_000/Dropbox/UVA 2016-2017/Spring Semester/CS 4501/project2/gt/gt.npy')

WIDTH = len(left[0])
HEIGHT = len(left)
MAX_DISPARITY = int(HEIGHT/3)

print ("Width of the image is: " + str(WIDTH))
print ("Height of the Image is: " + str(HEIGHT))
	
def ssd(pixel1, pixel2):
	val1 = pixel1[0]*0.333333 + pixel1[1]*0.333333 + pixel1[2]*0.3333333
	val2 = pixel2[0]*0.333333 + pixel2[1]*0.333333 + pixel2[2]*0.3333333
	return (val1-val2)**2

#constructing DSI matrix
DSI = []
for i in range (1, MAX_DISPARITY):
	matrix = numpy.ones((HEIGHT,WIDTH))
	for rownum in range(HEIGHT):
		for colnum in range(WIDTH):
			if (colnum-i>=0):
				matrix[rownum][colnum]=ssd(left[rownum][colnum], right[rownum][colnum-i])
	DSI.append(matrix)

#blur with Gaussian
blurred = []
# for i in range(0,MAX_DISPARITY-1):
# 	blurred.append(scipy.ndimage.filters.gaussian_filter(DSI[i], sigma=1))

# blurred2 = []
# bilateral filter
for i in range(0,MAX_DISPARITY-1):
	matrix = numpy.ones((HEIGHT, WIDTH))
	blurred.append(cv2.bilateralFilter(DSI[i].astype('float32'), 5, 75, 75))

#Aggregated min costs from all slices
depth = numpy.zeros((HEIGHT,WIDTH))
for rownum in range(HEIGHT):
	for colnum in range(WIDTH):
		minD = 1.0
		disparity = 1
		for i in range(0,MAX_DISPARITY-1):
			temp = blurred[i]
			if (temp[rownum][colnum]<minD):
				minD=temp[rownum][colnum]
				disparity = i
		depth[rownum][colnum] = disparity
pylab.imshow(depth)
pylab.show()

#Calculate RMS
SSE = 0
for rownum in range(HEIGHT):
	for colnum in range(WIDTH):
 		SSE+=(depth[rownum][colnum]-ground[rownum][colnum])**2
SSE/=(HEIGHT*WIDTH)
RMS = SSE**0.5
print ("RMS of aggregated disparity image is: " + str(RMS))

#left Right Consistency
DSI2 = []
for i in range (1, MAX_DISPARITY):
	matrix = numpy.ones((HEIGHT,WIDTH))
	for rownum in range(HEIGHT):
		for colnum in range(WIDTH):
			if (colnum+i<WIDTH):
				matrix[rownum][colnum]=ssd(right[rownum][colnum], left[rownum][colnum+i])
	DSI2.append(matrix)

blurred2 = []
for i in range(0,MAX_DISPARITY-1):
	blurred2.append(scipy.ndimage.filters.gaussian_filter(DSI2[i], sigma=1))

depth2 = numpy.zeros((HEIGHT,WIDTH))
for rownum in range(HEIGHT):
	for colnum in range(WIDTH):
		minD = 1.0
		disparity = 1
		for i in range(0,MAX_DISPARITY-1):
			temp = blurred2[i]
			if (temp[rownum][colnum]<minD):
				minD=temp[rownum][colnum]
				disparity = i
		depth2[rownum][colnum] = disparity
pylab.imshow(depth2)
pylab.show()

SSE = 0
for rownum in range(HEIGHT):
	for colnum in range(WIDTH):
 		SSE+=(depth2[rownum][colnum]-ground[rownum][colnum])**2
SSE/=(HEIGHT*WIDTH)
RMS = SSE**0.5
print("RMS of right-based aggregated disparity image is: " + str(RMS))

aggregator = numpy.zeros((HEIGHT,WIDTH))
occluded =0
for rownum in range(HEIGHT):
	for colnum in range(WIDTH):
		if (abs(depth[rownum][colnum]-depth2[rownum][colnum])<15):
			aggregator[rownum][colnum]=min(depth[rownum][colnum],depth2[rownum][colnum])
		else:
			occluded+=1
SSE = 0
for rownum in range(HEIGHT):
	for colnum in range(WIDTH):
		if(aggregator[rownum][colnum]!=0):
 			SSE+=(aggregator[rownum][colnum]-ground[rownum][colnum])**2

SSE/=((HEIGHT*WIDTH)-occluded)
RMS = SSE**0.5
print("RMS of left-right aggregated disparity image is: " + str(RMS))

pylab.imshow(aggregator)
pylab.show()
